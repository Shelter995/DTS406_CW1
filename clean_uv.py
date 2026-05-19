import os
import subprocess
import shutil


def run_deep_clean():
    # 你的 uv 全局缓存目录解压区
    cache_base = r"D:\Tools\uv\cache\archive-v0"

    print("=== 第一阶段：尝试调用 uv 官方指令 ===")
    print("正在执行: uv cache prune")
    # 先试试自带的指令能不能解决
    subprocess.run(["uv", "cache", "prune"], shell=True)
    print("-" * 50)

    print("=== 第二阶段：底层物理硬链接深度扫描 ===")
    if not os.path.exists(cache_base):
        print(f"未找到缓存目录: {cache_base}，请检查路径。")
        return

    orphans = []

    print("正在扫描哈希文件夹，这可能需要几秒钟...\n")
    # 遍历 archive-v0 下的所有哈希文件夹
    for hash_dir in os.listdir(cache_base):
        full_dir = os.path.join(cache_base, hash_dir)
        if not os.path.isdir(full_dir):
            continue

        # 获取包名 (读取 .dist-info 文件夹的名称，方便人类阅读)
        pkg_name = "未知包"
        for item in os.listdir(full_dir):
            if item.endswith(".dist-info"):
                pkg_name = item.replace(".dist-info", "")
                break

        # 核心逻辑：检查硬链接数
        in_use = False
        # 为了提高扫描效率，只要找到任何一个文件的硬链接数 > 1，就说明该库正被使用，立即跳过
        for root, _, files in os.walk(full_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # st_nlink 返回物理文件系统上的硬链接数
                    if os.stat(file_path).st_nlink > 1:
                        in_use = True
                        break
                except OSError:
                    pass
            if in_use:
                break

        # 如果遍历完发现没有任何文件的链接数 > 1，那就是彻底的孤儿
        if not in_use:
            # 计算这个包占用了多少空间
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(full_dir)
                for filename in filenames
            )
            size_mb = size_bytes / (1024 * 1024)
            orphans.append({
                "path": full_dir,
                "name": pkg_name,
                "size": size_mb,
                "hash": hash_dir
            })

    # === 第三阶段：报告与清理 ===
    if not orphans:
        print("✅ 完美！底层扫描确认：你的 uv 缓存中没有任何未关联的废弃库，空间利用率为 100%。")
        return

    print(f"🚨 扫描完毕！突破了官方指令的限制，揪出了 {len(orphans)} 个隐藏的孤儿库：")
    total_size = 0
    for orphan in orphans:
        print(f"  [闲置] {orphan['name']:<30} (哈希: {orphan['hash']}, 占用: {orphan['size']:>7.2f} MB)")
        total_size += orphan['size']

    print(f"\n🗑️ 总共可强制释放空间: {total_size:.2f} MB")

    # 最终确认是否使用 Python 的强制删除
    choice = input("\nuv 官方指令已忽略它们。是否要使用底层命令将这些顽固库彻底物理删除？(y/n): ")
    if choice.lower() == 'y':
        for orphan in orphans:
            try:
                # 相当于在终端执行 rd /s /q
                shutil.rmtree(orphan['path'])
            except Exception as e:
                print(f"删除 {orphan['name']} 失败，可能文件被占用: {e}")
        print(f"\n✅ 清理完成！成功为你夺回 {total_size:.2f} MB 的磁盘空间。")
    else:
        print("\n已取消删除操作。")


if __name__ == "__main__":
    run_deep_clean()