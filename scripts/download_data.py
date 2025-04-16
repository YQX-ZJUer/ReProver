"""Script to download LeanDojo Benchmark and LeanDojo Benchmark 4 into `./data`."""
# 从网上（Zenodo）下载数据集并生成本地data目录，这个脚本需要在训练之前手动执行
import os
import argparse
from hashlib import md5
from loguru import logger


LEANDOJO_BENCHMARK_4_URL = (
    "https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz?download=1"
)
DOWNLOADS = {
    LEANDOJO_BENCHMARK_4_URL: "25e1ee60cd8925b9d2e8673ddcc34b4c",
}


def check_md5(filename: str, gt_hashcode: str) -> bool:# md5哈希值验证文件是否完整，分块读取大文件并计算哈希值，避免内存溢出，确保文件完整性
    """
    Check the MD5 of a file against the ground truth.
    """
    if not os.path.exists(filename):
        return False
    # The file could be large.
    # See https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file.
    inp = open(filename, "rb")
    hasher = md5()
    while True:
        block = inp.read(64 * (1 << 20))
        if not block:
            break
        hasher.update(block)
    return hasher.hexdigest() == gt_hashcode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")# 这里默认目录是“data”
    args = parser.parse_args()
    logger.info(args)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    for url, hashcode in DOWNLOADS.items():
        logger.info(f"Downloading {url}")
        path = f"{args.data_path}/{os.path.basename(url)}"
        os.system(f"wget {url} -O {path}")
        if not check_md5(path, hashcode):
            raise RuntimeError(f"MD5 of {path} does not match the ground truth.")

        logger.info(f"Extracting {path}")
        os.system(f"tar -xf {path} -C {args.data_path}")

        logger.info(f"Removing {path}")
        os.remove(path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
