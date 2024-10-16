import os
from datetime import datetime
import shutil

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_directory_path = f'experiments/{now}'
    os.mkdir(new_directory_path)


    README_TEMPLATE = f"""
    
    # EXPERIMENT {now}

    # Training Data Notes

    # Model Training Notes

    # Application Notes

    """

    shutil.copytree('pointmouse', f'{new_directory_path}/pointmouse', dirs_exist_ok=True)
        

    with open(f"{new_directory_path}/README.md", "w") as f:
        f.write(README_TEMPLATE)