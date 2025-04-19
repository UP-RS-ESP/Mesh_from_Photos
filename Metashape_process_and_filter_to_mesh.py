# Load this file in the console - I turned off the extra output files of the various steps.
#
import Metashape
import sys, os
import numpy as np

nr_of_chunks = len(
    Metashape.app.document.chunks
)  # by default all chunks in the current workspace are processed
path2save = "./"  # path to save output data'


def removeSmallComponents(model, faces_threshold):
    model.removeComponents(faces_threshold)
    stats = model.statistics()
    print(
        "After removing small components with faces_threshold={}: {} faces in {} components left".format(
            faces_threshold, stats.faces, stats.components
        )
    )
    return stats


# first iteration only filters by image count
def filter1_sparse_pc(chunk, imgcount=2):
    # Filter sparse point cloud
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ImageCount)
    # np.savetxt(os.path.join(path2save,'ImageCount0.csv.gz'), f.values, delimiter=',')
    # print('imgcount': imgcount)
    # np.savetxt(os.path.join(path2save,'ImageCount0.thresold'), imgcount, delimiter=',')
    f.removePoints(imgcount)


# second filter step removes tie points above reprojection error 1
def filter2_sparse_pc(chunk):
    # Filter sparse point cloud
    ## ReprojectionError
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.csv.gz'), f.values, delimiter=',')
    # print('reperr': reperr)
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.thresold'), reperr, delimiter=',')
    f.removePoints(1)


# third filter step removes 90th percentile reprojection errors
def filter3_sparse_pc(chunk):
    # Filter sparse point cloud
    ## ReprojectionError
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    reperr_90p = np.percentile(f.values, 90)
    reperr = reperr_90p
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.csv.gz'), f.values, delimiter=',')
    # print('reperr': reperr)
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.thresold'), reperr, delimiter=',')
    f.removePoints(reperr)


# combined filtering steps will filter by 90th percentile
def filter_combined90p_sparse_pc(chunk):
    # Filter sparse point cloud
    ## ReprojectionError
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    reperr_90p = np.percentile(f.values, 90)
    reperr = reperr_90p
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.csv.gz'), f.values, delimiter=',')
    # print('reperr': reperr)
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.thresold'), reperr, delimiter=',')
    f.removePoints(reperr)

    ## ReconstructionUncertainty
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
    recunc_90p = np.percentile(f.values, 90)
    recunc = recunc_90p
    # np.savetxt(os.path.join(path2save,'ReconstructionUncertainty0.csv.gz'), f.values, delimiter=',')
    # print('recunc': recunc)
    # np.savetxt(os.path.join(path2save,'ReconstructionUncertainty0.thresold'), recunc, delimiter=',')
    f.removePoints(recunc)

    ## Image count
    # f = Metashape.TiePoints.Filter();
    # f.init(chunk, Metashape.TiePoints.Filter.ImageCount)
    # np.savetxt(os.path.join(path2save,'ImageCount0.csv.gz'), f.values, delimiter=',')
    # print('imgcount': imgcount)
    # np.savetxt(os.path.join(path2save,'ImageCount0.thresold'), imgcount, delimiter=',')
    # f.removePoints(imgcount)

    ##
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
    projacc_90p = np.percentile(f.values, 90)
    projacc = projacc_90p
    # np.savetxt(os.path.join(path2save,'ProjectionAccuracy0.csv.gz'), f.values, delimiter=',')
    # print('projacc': projacc)
    # np.savetxt(os.path.join(path2save,'ProjectionAccuracy0.thresold'), projacc, delimiter=',')
    f.removePoints(projacc)


# second filtering uses 95th percentile (could change to 90th percentile if the images are noisy)
def filter_compbined95p_sparse_pc(chunk, percentile=95):
    # Filter sparse point cloud

    ## ReprojectionError
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    reperr_95p = np.percentile(f.values, percentile)
    reperr = reperr_95p
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.csv.gz'), f.values, delimiter=',')
    # print('reperr': reperr)
    # np.savetxt(os.path.join(path2save,'ReprojectionError0.thresold'), reperr, delimiter=',')
    f.removePoints(reperr)

    ## ReconstructionUncertainty
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
    recunc_95p = np.percentile(f.values, percentile)
    recunc = recunc_95p
    # np.savetxt(os.path.join(path2save,'ReconstructionUncertainty0.csv.gz'), f.values, delimiter=',')
    # print('recunc': recunc)
    # np.savetxt(os.path.join(path2save,'ReconstructionUncertainty0.thresold'), recunc, delimiter=',')
    f.removePoints(recunc)

    # ## Image count - only needed in first iteration
    # f = Metashape.TiePoints.Filter();
    # f.init(chunk, Metashape.TiePoints.Filter.ImageCount)
    # # # np.savetxt(os.path.join(path2save,'ImageCount0.csv.gz'), f.values, delimiter=',')
    # # # print('imgcount': imgcount)
    # # # np.savetxt(os.path.join(path2save,'ImageCount0.thresold'), imgcount, delimiter=',')
    # f.removePoints(imgcount)

    ##
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
    projacc_95p = np.percentile(f.values, percentile)
    projacc = projacc_95p
    # np.savetxt(os.path.join(path2save,'ProjectionAccuracy0.csv.gz'), f.values, delimiter=',')
    # print('projacc': projacc)
    # np.savetxt(os.path.join(path2save,'ProjectionAccuracy0.thresold'), projacc, delimiter=',')
    f.removePoints(projacc)


def calculate_covariance_parameters(chunk):
    # see also: https://www.agisoft.com/forum/index.php?topic=11218.msg50653#msg50653
    print("calculate covariances in scaled local coordinates")
    T = chunk.transform.matrix
    if (
        chunk.transform.translation
        and chunk.transform.rotation
        and chunk.transform.scale
    ):
        T = chunk.crs.localframe(T.mulp(chunk.region.center)) * T
    R = T.rotation() * T.scale()
    nrp = len(chunk.tie_points.points)
    print("%d tie poits" % nrp)

    coords = np.empty((nrp, 3))
    coords.fill(np.nan)
    vect = np.empty((nrp, 3))
    vect.fill(np.nan)
    var = np.empty(nrp)
    var.fill(np.nan)
    track_ids = np.empty(nrp)
    track_ids.fill(np.nan)

    for i in range(nrp):
        point = chunk.tie_points.points[i]
        if not point.valid:
            continue
        cov = point.cov
        coord = point.coord

        foo = T * coord
        coords[i, :] = np.asarray(foo[0:3])
        cov = R * cov * R.t()
        u, s, v = cov.svd()
        var[i] = np.sqrt(np.sum(s))  # variance vector length
        vect[i, :] = np.array(u.col(0)) * var[i]
        track_ids[i] = point.track_id

    return track_ids, coords, var, vect


# MAIN
for i in range(nr_of_chunks):
    chunk = Metashape.app.document.chunks[i]
    dirname = chunk.label

    # Detect Markers and add Scalebars (assumes 0.12 m distance between markers)
    if len(chunk.markers) < 1:
        print("Detect Markers and add Scalebars")
        task_dm = Metashape.Tasks.DetectMarkers()
        # PhotoScan.TargetType.CircularTarget12bit, 50
        task_dm.apply(chunk)

        # add positions for first 4 points - this may need to be edited
        chunk.markers[0].reference.location = [0, 0, 0]
        chunk.markers[1].reference.location = [0.12, 0, 0]
        chunk.markers[2].reference.location = [0, -0.12, 0]
        chunk.markers[3].reference.location = [0.12, -0.12, 0]

        # add Scalebars based on boards
        nr_markers = len(chunk.markers)
        nr_of_markers_on_board = 6
        for j in range(int(nr_markers / nr_of_markers_on_board)):
            start_marker = j * nr_of_markers_on_board
            for k in range(5):
                if np.mod(k, 2) == 0 or k == 0:
                    sb = chunk.addScalebar(
                        chunk.markers[start_marker + k],
                        chunk.markers[start_marker + k + 1],
                    )
                else:
                    if k < 4:
                        sb = chunk.addScalebar(
                            chunk.markers[start_marker + k],
                            chunk.markers[start_marker + k + 2],
                        )
                sb.reference.distance = 0.12

    # if not already done, match photos and align cameras
    aligned = [
        camera
        for camera in chunk.cameras
        if camera.transform and camera.type == Metashape.Camera.Type.Regular
    ]
    if aligned == []:
        print("photos not aligned yet")
        task_mp = Metashape.Tasks.MatchPhotos()
        task_mp.downscale = 1  # Work with photos at their original size
        task_mp.guided_matching = True
        task_mp.keypoint_limit_per_mpx = 4000
        task_mp.tiepoint_limit = 20000
        task_mp.generic_preselection = True
        task_mp.reference_preselection = True
        task_mp.reset_matches = True
        task_mp.subdivide_task = True
        task_mp.filter_stationary_points = True
        task_mp.apply(chunk)
        chunk.alignCameras()
        # Metashape.Document.save()
        Metashape.app.document.save()

    # Optimize Cameras and filter
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter1_sparse_pc(chunk)

    # Second iteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter2_sparse_pc(chunk)

    # Third iteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter3_sparse_pc(chunk)

    # Fourth iteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter_combined90p_sparse_pc(chunk)

    # Fifth iteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter_combined90p_sparse_pc(chunk)

    # Sixth iteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter_combined90p_sparse_pc(chunk)

    # Seventhiteration of camera optimization
    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    filter_combined90p_sparse_pc(chunk)

    chunk.optimizeCameras(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_b1=False,
        fit_b2=False,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=False,
        fit_p1=True,
        fit_p2=True,
        fit_corrections=False,
        adaptive_fitting=True,
        tiepoint_covariance=True,
    )
    print("Calculate variance and covariance for every tie point (SIFT feature)")
    track_ids, coords, var, vect = calculate_covariance_parameters(chunk)
    #
    # fname = os.path.join(path2save, '%s_tiepoints_covariances2_f.csv.gz'%dirname)
    # print('Saving Projection uncertainties and covariance to %s'%fname)
    # np.savetxt(fname, np.c_[track_ids, coords, var, vect], delimiter=',',
    #     header='trackid, x, y, z, var, cov_x, cov_y, cov_z',  fmt='%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' )

    # Get all values for filtering and store to matrix
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    reperr_values = f.values.copy()
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
    recunc_values = f.values.copy()
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ImageCount)
    imgcount_values = f.values.copy()
    f = Metashape.TiePoints.Filter()
    f.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
    projacc_values = f.values.copy()

    fname = os.path.join(path2save, "%s_tiepoints_covariances_errors.csv.gz" % dirname)
    print("Saving Projection uncertainties and covariance to %s" % fname)
    np.savetxt(
        fname,
        np.c_[
            track_ids,
            coords,
            var,
            vect,
            reperr_values,
            recunc_values,
            imgcount_values,
            projacc_values,
        ],
        delimiter=",",
        header="trackid, x, y, z, var, cov_x, cov_y, cov_z, reprojection_error, reconstruction_uncertainty, image_count, projection_accuracy",
        fmt="%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %d, %.5f",
    )

    crs = Metashape.CoordinateSystem(
        'LOCAL_CS["Local Coordinates",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'
    )

    fname = os.path.join(path2save, "%s_camera_locations.csv" % dirname)
    chunk.exportReference(
        path=fname,
        format=Metashape.ReferenceFormatCSV,
        items=Metashape.ReferenceItemsCameras,
        columns="nuvwpqrdefijk",
        # n - label, o - enabled flag, x/y/z - coordinates, X/Y/Z - coordinate accuracy, a/b/c - rotation angles,
        # A/B/C - rotation angle accuracy, u/v/w - estimated coordinates, U/V/W - coordinate errors,
        # d/e/f - estimated orientation angles, D/E/F - orientation errors,
        # p/q/r - estimated coordinates variance, i/j/k - estimated orientation angles variance,
        # [] - group of multiple values, | - column separator within group
        delimiter=",",
        precision=6,
    )

    #     fname = os.path.join(path2save, "%s_cameras.csv" % dirname)
    #     chunk.exportCameras(
    #         path=fname,
    #         format=CamerasFormatXML,
    #         crs=crs,
    #         save_points=True,
    #         save_markers=False,
    #         save_invalid_matches=False,
    #         save_absolute_paths=False,
    #         use_labels=False,
    #         use_initial_calibration=False,
    #         image_orientation=0,
    #         chan_rotation_order=RotationOrderXYZ,
    #         binary=False,
    #         bundler_save_list=True,
    #         bundler_path_list="list.txt",
    #         bingo_save_image=True,
    #         bingo_save_itera=True,
    #         bingo_save_geoin=True,
    #         bingo_save_gps=False,
    #         bingo_path_itera="itera.dat",
    #         bingo_path_image="image.dat",
    #         bingo_path_geoin="geoin.dat",
    #         bingo_path_gps="gps-imu.dat",
    #     )
    Metashape.app.document.save()

    # chunk.exportPoints("d:\test.ply", source = PhotoScan.DataSource.PointCloudData, format = PhotoScan.PointsFormatPLY, crs=crs)
    # chunk.exportReport(output_folder + '/report.pdf')

    if not chunk.depth_maps:
        print("build depth maps for %s" % dirname)
        # full resolution downscale = 1 will be much slower
        chunk.buildDepthMaps(
            downscale=1,
            max_neighbors=32,
            workitem_size_cameras=40,
            filter_mode=Metashape.AggressiveFiltering,
        )
        Metashape.app.document.save()

    if not chunk.models:
        print("build 3D model from depth maps for %s" % dirname)
        # For the point cloud based reconstruction they are calculated based on the number of points in the source point cloud:
        # the ratio is 1/5, 1/15, and 1/45 respectively.
        # face_count depends in input density. HighFaceCount using 1/5 of the available points - results in around 20e6 points
        chunk.buildModel(
            surface_type=Metashape.Arbitrary,
            interpolation=Metashape.EnabledInterpolation,
            face_count_custom=0,
            face_count=Metashape.CustomFaceCount,  # face_count=Metashape.HighFaceCount,
            source_data=Metashape.DepthMapsData,
            vertex_colors=True,
            vertex_confidence=True,
            volumetric_masks=False,
            keep_depth=True,
            trimming_radius=10,
            subdivide_task=True,
            workitem_size_cameras=40,
            max_workgroup_size=100,
        )
        # face_count_custom=200000,

        # remove small components, keep only largest component - assumes there is one very large component
        # https://github.com/agisoft-llc/metashape-scripts/blob/master/src/contrib/gradual_selection_mesh.py
        print("filtering 3D model %s" % dirname)
        stats = chunk.model.statistics()
        print(
            "Model has {} faces in {} components".format(stats.faces, stats.components)
        )

        while stats.components > 1:
            component_avg_faces = np.ceil(stats.faces / stats.components)
            # the largest component is for sure bigger then average component faces number, so we will filter only small components
            # probably there are some small components left - so we will continue our while-loop if needed
            faces_threshold = int(component_avg_faces)
            new_stats = removeSmallComponents(chunk.model, faces_threshold)

            assert (
                new_stats.components < stats.components
            )  # checking that we deleted at least the smallest something (to ensure that script works fine)
            assert (
                new_stats.components > 0
            )  # checking that the largest component is still there (to ensure that script works fine)
            stats = new_stats

        chunk.model.fixTopology()
        # stats = chunk.model.statistics()
        print(
            "Model has {} faces in {} components after filtering".format(
                stats.faces, stats.components
            )
        )
        # you may need to clip the area first before decimating and smoothing
        chunk.decimateModel(face_count=10000000)
        chunk.smoothModel(strength=3)

        # print('build 3D tiled model from depth maps for %s'%dirname)
        # pixel size is in m - using 0.1 mm
        # tile_size is in pixels. With 0.1 mm, you can use tile_size=8192 to obtain 0.8192 m tiles
        # we aim for 100,000 face_counts (20,000 is default) = 42 MP * 100,000
        # Number of faces per megapixel of texture resolution.
        # chunk.buildTiledModel(pixel_size=0.0001, tile_size=4096, source_data=Metashape.DepthMapsData, face_count=100000,
        #    ghosting_filter=False, transfer_texture=False, keep_depth=True, merge=False,
        #    subdivide_task=True, workitem_size_cameras=20, max_workgroup_size=100)

    Metashape.app.document.save()
    mesh_fname = os.path.join(
        path2save, "./%s_uhq_mesh_ds1_aggf_10M_faces_smooth.ply" % dirname
    )
    if not os.path.exists(mesh_fname):
        print("export decimated 10M faces smoothed, no-texture mesh %s" % mesh_fname)
        chunk.exportModel(
            path=mesh_fname,
            binary=True,
            precision=6,
            save_texture=False,
            save_uv=False,
            save_normals=True,
            save_colors=True,
            save_confidence=False,
            save_cameras=False,
            save_markers=False,
            save_udim=False,
            save_alpha=False,
            embed_texture=False,
            strip_extensions=False,
            raster_transform=Metashape.RasterTransformNone,
            colors_rgb_8bit=True,
            comment="",
            save_comment=True,
            format=Metashape.ModelFormatNone,
            crs=crs,
            clip_to_boundary=True,
            save_metadata_xml=False,
        )

    # if not chunk.model.textures:
    #     print('build texture model from depth maps for %s'%dirname)
    #     chunk.buildUV(texture_size=16384, page_count=1)
    #     chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=16384, fill_holes=True, ghosting_filter=True,
    #         texture_type=Metashape.Model.DiffuseMap, transfer_texture=True)
    #     mesh_fname = './%s_hq_mesh_ds2_mildf_100e6_faces_texture_diffuse.ply'%dirname
    #     print('export mesh %s'%mesh_fname)
    #     chunk.exportModel(path=mesh_fname, binary=True, precision=6, texture_format=Metashape.ImageFormatJPEG, save_texture=True,
    #         save_uv=True, save_normals=True, save_colors=True, save_confidence=True,
    #         save_cameras=False, save_markers=False, save_udim=False, save_alpha=False,
    #         embed_texture=True, strip_extensions=False, raster_transform=Metashape.RasterTransformNone,
    #         colors_rgb_8bit=True, comment='', save_comment=True, format=Metashape.ModelFormatNone, crs=crs,
    #         clip_to_boundary=True, save_metadata_xml=False)
    #
    #     chunk.buildTexture(texture_size=16384, fill_holes=False, ghosting_filter=False,
    #         texture_type=Metashape.Model.OcclusionMap, source_model=chunk.model, transfer_texture=False)
    #     mesh_fname = './%s_hq_mesh_ds2_mildf_100e6_faces_texture_occlusion.ply'%dirname
    #     print('export mesh %s'%mesh_fname)
    #     chunk.exportModel(path=mesh_fname, binary=True, precision=6, texture_format=Metashape.ImageFormatJPEG, save_texture=True,
    #         save_uv=True, save_normals=True, save_colors=True, save_confidence=True,
    #         save_cameras=False, save_markers=False, save_udim=False, save_alpha=False,
    #         embed_texture=True, strip_extensions=False, raster_transform=Metashape.RasterTransformNone,
    #         colors_rgb_8bit=True, comment='', save_comment=True, format=Metashape.ModelFormatNone, crs=crs,
    #         clip_to_boundary=True, save_metadata_xml=False)

    Metashape.app.document.save()
