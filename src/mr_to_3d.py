import open3d as o3d
import numpy as np
import os

if __name__ == "__main__":
    data_path = "../data/MRNet/MRNet-v1.0/train/axial"
    filelist = os.listdir(data_path)
    if '.DS_Store' in filelist:
        filelist.remove('.DS_Store')

    filename = filelist[0]
    filepath = os.path.join(data_path, filename)
    img = np.load(filepath)
    img = img.astype(float)

    img /= img.max()
    img_mask = img > 0.5
    points = np.argwhere(img_mask).astype(float)
    points[:, 0] /= points[:, 0].max()
    points[:, 1] /= points[:, 1].max()
    points[:, 2] /= points[:, 2].max()
    print(points.shape)

    # downsample option 2: after intensity thresholding
    points = points[::100]
    print(points.shape)

    #point_colors = np.ones_like(points)
    #point_colors *= img[img_mask][:, np.newaxis]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries([pcd])
