import os
import os.path as osp
import shutil
from glob import glob

LIGER_LOC = "/home/remi_ouazan/Liger-Kernel"
FA_DIR_IN_LIGER = "src/liger_kernel/ops/flash_attention"

if __name__ == "__main__":

    for file in glob("src/**.py") + glob("src/**/*.py"):

        # # Clean destination
        # shutil.rmtree(osp.join(LIGER_LOC, FA_DIR_IN_LIGER))

        # Skip other implementation
        if "other_implementations" in file:
            continue

        # Infer and guaranty dst directory
        dst_file = osp.join(LIGER_LOC, file.replace("src", FA_DIR_IN_LIGER))
        dst_directory = osp.dirname(dst_file)
        os.makedirs(dst_directory, exist_ok=True)

        # Get the file in a buffer and replace some stuff
        with open(file) as f:
            buffer = f.read()
        buffer = buffer.replace("from src.", "from " + FA_DIR_IN_LIGER.replace("/", ".") + ".")

        # Write it back
        with open(dst_file, "w") as f:
            f.write(buffer)

        print(dst_file)
