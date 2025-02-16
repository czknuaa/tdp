import torch

def kmeans_gpu(points, num_clusters, num_iters=100, tol=1e-4):
    """
    使用 PyTorch 在 GPU 上实现 KMeans 聚类算法，支持批量数据
    :param points: 输入的点云数据，形状为 (B, N, D)，B 是批次大小，N 是点的数量，D 是每个点的维度
    :param num_clusters: 聚类的簇数量
    :param num_iters: 最大迭代次数
    :param tol: 目标函数收敛的阈值
    :return: 聚类标签和聚类中心
    """
    B, N, D = points.shape  # B 是批次大小，N 是点的数量，D 是每个点的维度

    # 初始化簇中心（随机选择 K 个点作为初始簇中心）
    centroids = points[:, torch.randperm(N)[:num_clusters], :]  # 形状为 (B, K, D)

    # 聚类标签初始化
    labels = torch.zeros(B, N, device=points.device, dtype=torch.long)

    # 用于保存前一个簇中心
    old_centroids = centroids.clone()

    for i in range(num_iters):
        # 计算每个点到簇中心的距离 (B, N, K)，这里的广播机制会自动扩展维度
        distances = torch.cdist(points.view(-1, D), centroids.view(-1, D))  # (B * N, K)
        
        # 将计算结果从 (B * N, K) 变形为 (B, N, K)
        distances = distances.view(B, N, num_clusters)  # 形状为 (B, N, K)

        # 为每个点分配最接近的簇中心
        new_labels = torch.argmin(distances, dim=2)  # (B, N)，每个点的簇标签

        # 如果标签没有改变，提前结束迭代
        if torch.all(new_labels == labels):
            break

        labels = new_labels

        # 更新簇中心
        for b in range(B):  # 遍历批次
            for j in range(num_clusters):
                # 获取该批次下所有属于簇 j 的点
                cluster_points = points[b, labels[b] == j]
                if cluster_points.shape[0] > 0:
                    centroids[b, j] = cluster_points.mean(dim=0)

        # 如果簇中心变化小于 tol，认为已收敛
        if torch.norm(centroids - old_centroids, p=2) < tol:
            break

        # 更新 old_centroids 为当前的 centroids
        old_centroids = centroids.clone()

    return labels, centroids

# 测试示例
if __name__ == "__main__":
    # 随机生成 10 个批次，每个批次 1000 个 3D 点
    B = 10  # 批次大小
    N = 1000  # 每批次点云中的点数量
    D = 3  # 点的维度（3D 点）
    points = torch.randn(B, N, D).cuda()  # 点云数据，移动到 GPU 上
    
    num_clusters = 5  # 选择 5 个簇
    labels, centroids = kmeans_gpu(points, num_clusters)

    print(f"聚类标签: {labels}")
    print(f"聚类中心: {centroids}")
