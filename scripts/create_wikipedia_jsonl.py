import os
import json 
image_root = "wikipedia_images_full"
input_file = "Wiki6M_ver_1_0.jsonl"
output_file = "Wiki6M_ver_1_0_updated.jsonl"

# Process the file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        wikidata_id = data.get("wikidata_id")

        if wikidata_id:
            # Build the image path
            folder = wikidata_id[:4]
            image_filename = f"{wikidata_id}.jpg"
            image_path = os.path.join(image_root, folder, image_filename)

            # Check if the image file exists
            if os.path.exists(image_path):
                data["image_path"] = image_path
                outfile.write(json.dumps(data) + "\n")


print("JSONL file updated with image paths!")
