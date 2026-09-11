from PIL import Image
import pathlib as pl
import os

# Originally made to split the large flatmount tile scan into 512x512 ROIs for REShAPE CPSAM training
def split_folder_into_rois(input_path, output_path, roi_size, image_extensions):
    flatmount_dir = os.listdir(input_path)
    ext = tuple(ext.lower() for ext in image_extensions)

    image_files = [os.path.join(input_path, f) for f in flatmount_dir if any(f.lower().endswith(x) for x in ext)]

    total_tiles = 0
    for i in image_files:
        print(f"Processing: {i}")
        img = Image.open(i)
        width, height = img.size
        base_name, file_type = os.path.splitext(os.path.basename(i))

        tile_index = 1
        for top in range(0, height - roi_size + 1, roi_size):
            for left in range(0, width - roi_size + 1, roi_size):
                box = (left, top, left + roi_size, top + roi_size)
                roi = img.crop(box)

                if roi.size == (roi_size, roi_size):
                    suffix = f"_{roi_size}px2_tile_{tile_index:02d}"
                    out = f"{output_path}\\{base_name}{suffix}{file_type}"

                    roi.save(out)

                    tile_index += 1
                    total_tiles += 1

            print(f"Split image {base_name}{file_type} into {tile_index} {roi_size}px^2 tiles")

    print(f"{len(image_files)} tile scan images processed. {total_tiles} {roi_size}px^2 tiles saved to {output_path}")

if __name__ == "__main__":
    split_folder_into_rois(
        input_path = str(pl.Path(r'/data_flatmount_images')),
        output_path = str(pl.Path(r'/data_flatmount_images\ROI_outputs')),
        roi_size = 512,
        image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )