from plyfile import PlyData

def preview_ply_vertices(path: str, n: int = 5):
    """
    使用 plyfile 读入 PLY，然后打印前 n 条顶点属性（x,y,z,…）。
    对 binary_little_endian 格式同样适用。
    """
    ply = PlyData.read(path)
    vertex_data = ply['vertex'].data  # numpy structured array

    # 打印字段名
    fields = vertex_data.dtype.names
    print("Fields:", fields)
    print()

    # 打印前 n 条记录
    for i, v in enumerate(vertex_data[:n]):
        print(f"#{i:02d}:", tuple(v[field] for field in fields))

preview_ply_vertices("/users/lshou/4DGaussians/output/dnerf/jumpingjacks/gaussian_interpolated/time_00100.ply", n=10)