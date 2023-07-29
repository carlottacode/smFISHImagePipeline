from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.morphology import binary_opening

def _get_distance_cell(cell_mask):
    # build distance map from the cell boundaries
    distance_cell = ndi.distance_transform_edt(cell_mask)
    distance_cell = distance_cell.astype(np.float32)
    distance_cell_normalized = distance_cell / distance_cell.max()
    return distance_cell

def features_distance(
        rna_coord,
        distance_cell,
        cell_mask,
        ndim,
        channel,
        distance_nuc=None,
        check_input=True):
    """Compute distance related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    distance_cell : np.ndarray, np.float32
        Distance map from the cell with shape (y, x).
    distance_nuc : np.ndarray, np.float32
        Distance map from the nucleus with shape (y, x).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_mean_dist_cell : float
        Normalized mean distance of RNAs to the cell membrane.
    index_median_dist_cell : float
        Normalized median distance of RNAs to the cell membrane.
    index_mean_dist_nuc : float
        Normalized mean distance of RNAs to the nucleus.
    index_median_dist_nuc : float
        Normalized median distance of RNAs to the nucleus.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(
            rna_coord,
            ndim=2,
            dtype=[np.int32, np.int64])
        stack.check_array(
            distance_cell,
            ndim=2,
            dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = (1., 1.)
        return features, ['index_mean_dist_cell_'+channel, 'index_median_dist_cell_'+channel]

    # compute mean and median distance to cell membrane
    rna_distance_cell = distance_cell[
        rna_coord[:, ndim - 2],
        rna_coord[:, ndim - 1]]
    expected_distance = np.mean(distance_cell[cell_mask])
    index_mean_dist_cell = np.mean(rna_distance_cell) / expected_distance
    expected_distance = np.median(distance_cell[cell_mask])
    index_median_dist_cell = np.median(rna_distance_cell) / expected_distance

    features = (index_mean_dist_cell, index_median_dist_cell)

    return features, ['index_mean_dist_cell_'+channel, 'index_median_dist_cell_'+channel]



def features_protrusion(
        rna_coord,
        cell_mask,
        ndim,
        voxel_size_yx,
        channel,
        nuc_mask=None,
        check_input=True):
    """Compute protrusion related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_rna_protrusion : float
        Number of RNAs detected in a protrusion and normalized by the expected
        number of RNAs under random distribution.
    proportion_rna_protrusion : float
        Proportion of RNAs detected in a protrusion.
    protrusion_area : float
        Protrusion area (in pixels).

    """
    # TODO fin a better feature for the protrusion (idea: dilate region from
    #  centroid and stop when a majority of new pixels do not belong to the
    #  cell).
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(
            ndim=int,
            voxel_size_yx=(int, float))
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # get number of rna and cell area
    nb_rna = len(rna_coord)
    cell_area = cell_mask.sum()

    # apply opening operator (3000 nanometers) and count the loss of RNAs
    size = int(3000 / voxel_size_yx)
    s = disk(size, dtype=bool)
    mask_cell_opened = binary_opening(cell_mask, selem=s)
    mask_cell_opened[nuc_mask] = True
    protrusion_area = cell_area - mask_cell_opened.sum()

    # case where we do not detect any
    if nb_rna == 0:
        features = (1., 0., protrusion_area)
        return features, ['index_rna_protrusion_'+channel, 'proportion_rna_protrusion_'+channel, 'protrusion_area_'+channel]

    if protrusion_area > 0:
        expected_rna_protrusion = nb_rna * protrusion_area / cell_area
        mask_rna = mask_cell_opened[
            rna_coord[:, ndim - 2],
            rna_coord[:, ndim - 1]]
        rna_after_opening = rna_coord[mask_rna]
        nb_rna_protrusion = nb_rna - len(rna_after_opening)
        index_rna_protrusion = nb_rna_protrusion / expected_rna_protrusion
        proportion_rna_protrusion = nb_rna_protrusion / nb_rna

        features = (
            index_rna_protrusion,
            proportion_rna_protrusion,
            protrusion_area)
    else:
        features = (1., 0., 0.)

    return features, ['index_rna_protrusion_'+channel, 'proportion_rna_protrusion_'+channel, 'protrusion_area_'+channel]

def features_dispersion(
        smfish,
        rna_coord,
        centroid_rna,
        cell_mask,
        centroid_cell,
        ndim,
        channel,
        check_input=True,
        centroid_nuc=None):
    """Compute RNA Distribution Index features (RDI) described in:

    RDI Calculator: An analysis Tool to assess RNA distributions in cells,
    Stueland M., Wang T., Park H. Y., Mili, S., 2019.

    Parameters
    ----------
    smfish : np.ndarray, np.uint
        Image of RNAs, with shape (y, x).
    rna_coord : np.ndarray, np.int
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    centroid_rna : np.ndarray, np.int
        Coordinates of the rna centroid with shape (2,) or (3,).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    centroid_cell : np.ndarray, np.int
        Coordinates of the cell centroid with shape (2,).
    centroid_nuc : np.ndarray, np.int
        Coordinates of the nucleus centroid with shape (2,).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_polarization : float
        Polarization index (PI).
    index_dispersion : float
        Dispersion index (DI).
    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(smfish, ndim=2, dtype=[np.uint8, np.uint16])
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(centroid_rna, ndim=1, dtype=[np.int32, np.int64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)
        stack.check_array(centroid_cell, ndim=1, dtype=[np.int32, np.int64])

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = (0., 1.)
        return features, ['index_polarization_'+channel, 'index_dispersion_'+channel]


    # initialization
    if ndim == 3:
        centroid_rna_2d = centroid_rna[1:]
    else:
        centroid_rna_2d = centroid_rna.copy()

    # get coordinates of each pixel of the cell
    cell_coord = np.nonzero(cell_mask)
    cell_coord = np.column_stack(cell_coord)

    # get coordinates of each rna pixel in the cell from a 2-d binary mask
    rna_mask = np.zeros_like(cell_mask)
    rna_mask[rna_coord[:, ndim - 2], rna_coord[:, ndim - 1]] = True
    rna_coord_ = np.nonzero(rna_mask)
    rna_coord_ = np.column_stack(rna_coord_)

    # get intensity value of every rna and cell pixels
    rna_value = smfish[rna_mask]
    total_intensity_rna = rna_value.sum()
    cell_value = smfish[cell_mask]
    total_intensity_cell = cell_value.sum()

    # compute polarization index from cell centroid
    centroid_distance = np.linalg.norm(centroid_rna_2d - centroid_cell)
    gyration_radius = _rmsd(cell_coord, centroid_cell)
    index_polarization = centroid_distance / gyration_radius

    features = (index_polarization,)

    # compute dispersion index
    r = np.linalg.norm(rna_coord_ - centroid_rna_2d, axis=1) ** 2
    a = np.sum((r * rna_value) / total_intensity_rna)
    r = np.linalg.norm(cell_coord - centroid_rna_2d, axis=1) ** 2
    b = np.sum((r * cell_value) / total_intensity_cell)
    index_dispersion = a / b

    features += (index_dispersion,)


    return features, ['index_polarization_'+channel, 'index_dispersion_'+channel]



def _rmsd(coord, reference_coord):
    """Compute the root-mean-squared distance between coordinates and a
    reference coordinate.

    Parameters
    ----------
    coord : np.ndarray, np.int
        Coordinates with shape (nb_points, 2).
    reference_coord : np.ndarray, np.int64
        Reference coordinate to compute the distance from, with shape (2,).

    Returns
    -------
    rmsd : float
        Root-mean-squared distance.

    """
    # compute RMSD between 'coord' and 'reference_coord'
    n = len(coord)
    diff = coord - reference_coord
    rmsd = float(np.sqrt((diff ** 2).sum() / n))

    return rmsd


def features_foci(rna_coord, foci_coord, ndim, check_input=True):
    """Compute foci related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    foci_coord : np.ndarray, np.int
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx coordinates), the number of
        spots detected in the foci and its index.
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    proportion_rna_in_foci : float
        Proportion of RNAs detected in a foci.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(foci_coord, ndim=2, dtype=[np.int32, np.int64])

    if len(rna_coord) == 0 or len(foci_coord) == 0:
        features = (0.,)
        return features, ['proportion_rna_in_foci']

    # compute proportion RNAs in foci
    nb_rna = len(rna_coord)
    nb_rna_in_foci = foci_coord[:, ndim].sum()
    proportion_rna_in_foci = nb_rna_in_foci / nb_rna

    features = (proportion_rna_in_foci,)

    return features, ['proportion_rna_in_foci']


def features_area(cell_mask, cell_mask_out_nuc=None, check_input=True, nuc_mask=None):
    """Compute area related features.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    check_input : bool
        Check input validity.

    Returns
    -------
    nuc_relative_area : float
        Proportion of nucleus area in the cell.
    cell_area : float
        Cell area (in pixels).
    nuc_area : float
        Nucleus area (in pixels).
    cell_area_out_nuc : float
        Cell area outside the nucleus (in pixels).
    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # get area of the cell and the nucleus
    cell_area = float(cell_mask.sum())

    # return features
    features = (cell_area,)

    return features,  ['cell_area']


def _get_centroid_rna(rna_coord, ndim):
    """Get centroid coordinates of RNA molecules.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    ndim : int
        Number of spatial dimensions to consider (2 or 3).

    Returns
    -------
    centroid_rna : np.ndarray, np.int
        Coordinates of the rna centroid with shape (2,) or (3,).

    """
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :ndim], axis=0, dtype=rna_coord.dtype)

    return centroid_rna
