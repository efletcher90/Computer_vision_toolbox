import xmltodict
import os
import re
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from mp_oct_enums_helpers import _to_uint8, _stim_text_hex

class CreateMicroperimetryDataDicts:
    def __init__(self, mp_dir, metadata_keys):
        self.mp_dir = mp_dir
        self.metadata_keys = metadata_keys

    # Create list of 'projection.xml' microperimetry datafiles
    def get_mp_list(self):
        mp_list = []
        seen_exams = set()

        # find all the scan folders within the root directory, sorted by visit number (e.g. V1 to V5).
        # each export holds all exams up to that visit, so an exam's own visit is the first folder it appears in
        folders = sorted([os.path.join(self.mp_dir, name) for name in os.listdir(self.mp_dir) if
                          os.path.isdir(os.path.join(self.mp_dir, name))],
                         key=lambda f: int(re.search(r'_V(\d+)', os.path.basename(f)).group(1)))

        # get the patient id from the file basename, just split at the first underscore. This is then assigned to the final dict
        p_id = os.path.basename(folders[-1]).split("_")[0]
        for f in folders:
            for dirpath, _, filenames in os.walk(f):
                # gather lower case filenames in folder
                filenames = {files.lower(): files for files in filenames}

                if 'projection.xml' in filenames:
                    # exam folder is named with the ExamID. skip it if already found in an earlier visit
                    exam_id = os.path.basename(dirpath)
                    if exam_id in seen_exams:
                        continue
                    seen_exams.add(exam_id)

                    # get timepoint from the path, e.g. ...\V5-Month18\... is 'Month18'
                    match = re.search(r'_V\d+-([^\\/]+)', dirpath)
                    visit_timepoint = match.group(1) if match else None

                    xml_path = os.path.join(dirpath, filenames['projection.xml'])

                    png_name = filenames.get('stillpicture.png')
                    png_path = os.path.join(dirpath, png_name)

                    mp_list.append({'xml_file': xml_path, 'scan_img': png_path, 'timepoint': visit_timepoint})

        return mp_list, p_id

    def parse_mp_xml_data(self, xml_files, p_id):
        all_xml_data = []

        for x in xml_files:
            xml_path = x['xml_file']
            png_path = x['scan_img']
            timepoint = x['timepoint']

            xml_dict = {}
            stim_dict = {}

            with open(xml_path, "rb") as f:
                # read xml as raw bytes ("rb"), then convert to dict w/ xmltodict
                raw_bytes = f.read()
                mp_data = xmltodict.parse(raw_bytes, force_list=("Stimulus",))

                # parsing the .xml structure w/ xmltodict module. It creates stim_key=stimulus_id and stim_value=(final_intensity, ray, angle_deg)
                stimulus_data = mp_data["MitProjection"]["Stimuli"]["Stimulus"]
                for stimulus_id in stimulus_data:
                    # need to add a zero to stimulus ids lower than 10, so that sorted() works properly
                    stim_key = f"stimulus_{int(stimulus_id['@id']):02d}"
                    # values are a list of intensity, ray and angle_deg, in order.
                    # Cartesian positioning used to remap the coordinates from ray and angle_deg
                    stim_value = [int(stimulus_id["@final_intensity"]), float(stimulus_id["@ray"]),
                                  float(stimulus_id["@angle_deg"])]
                    stim_dict[stim_key] = stim_value

            sorted_stim_dict = dict(sorted(stim_dict.items()))
            m_data = mp_data["MitProjection"]["Header"]

            # need to filter out any failed scans that have less than the 68 used in the microperimetry SOP
            n_stimuli = len([key for key in stim_dict if key != 'stimulus_00'])
            if n_stimuli < 68:
                continue

            sorted_stim_dict = dict(sorted(stim_dict.items()))

            # add patient metadata k:v pairs to xml_dict
            xml_dict.update({'patient_id': p_id, 'scan_path': png_path, 'timepoint':timepoint})
            xml_dict.update({k: m_data[k] for k in self.metadata_keys if k in m_data})
            # the sorted stim data is added after the patient metadata so that the original order of the enum list is kept at the start of the dict
            xml_dict.update(sorted_stim_dict)

            all_xml_data.append(xml_dict)

        return all_xml_data

class CalculateMicroperimetryCoordinates:
    def __init__(self, patient_data):
        self.patient_data = patient_data

    def calculate_pixel_coordinates(self):
        for patient in self.patient_data:
            # check and find dims of scan image (these should be 1536x1536 pixels - i.e. typical b-scan dims)
            img_path = patient["scan_path"]

            with Image.open(img_path) as img:
                width, height = img.size

                # using the fieldDegreesX/Y values from the xml, get the image scale
                scale_x = width / float(patient['fieldDegreesX'])
                scale_y = height / float(patient['fieldDegreesY'])

                # find grid centre then...
                # OS images appear to have the grid sitting at the 'baseline PRL' (0,0 for baseline exams). OD not so.
                centre_x, centre_y = width / 2.0, height / 2.0
                if str(patient['Eye']).strip().lower() in ('left', 'l', 'os'):
                    centre_x += float(patient["baselinePRLX_pix"])
                    centre_y += float(patient["baselinePRLY_pix"])

            # apply the manual centre offset, if used... seemingly not used in the xml files I have
            centre_x += float(patient["fixationCenterX"]) * scale_x
            centre_y -= float(patient["fixationCenterY"]) * scale_y

            # loop through the stimulus IDs and corresponding intensities and define their x/y pixels based on ray and angle_deg values.
            # intensities, ray and angle_deg are stored in stimulus_xx dicts as a list (in that order).
            # when calculated, values saved back to dict as list for unpacking/overlaying on png

            computed_stim_values = []
            for stim_keys, stim_vals in patient.items():
                if not stim_keys.startswith('stimulus_'):
                    continue

                # found out that data for stimulus_00 is actually not used and is the measure of the ONH
                if stim_keys == "stimulus_00":
                    continue

                # unpack the three data points from the nested list for each stim_key instance
                final_intensity, ray, angle_deg = stim_vals

                # MAIA stores thresholds as negative values. Also, a positive value of 1 means not seen
                intensity = -1 if final_intensity > 0 else -final_intensity
                ray = float(ray)
                angle_rad = math.radians(float(angle_deg))

                # had to check online what to do with the radians etc for the coordinate location, will check this is right
                px_x = centre_x + (ray * math.cos(angle_rad) * scale_x)
                px_y = centre_y - (ray * math.sin(angle_rad) * scale_y)

                computed_stim_values.append({
                    "id": stim_keys,
                    "intensity": intensity,
                    "px_x": px_x,
                    "px_y": px_y,
                })
            patient['computed_stim_values'] = computed_stim_values

        return self.patient_data

class FundusOverlays:
    def __init__(self, patient_data, colour_map, mp_dir, output_dir, folder_timestamp, aavb_font):
        self.patient_data = patient_data
        self.colour_map = colour_map
        self.mp_dir = mp_dir
        self.output_dir = output_dir
        self.timestamp = folder_timestamp
        self.font = ImageFont.truetype(aavb_font, 36)

    def draw_mp_data_to_maia_scan(self):
        # timestamp the output folder for reconstruction
        save_dir = os.path.join(self.output_dir, f"Untransformed_MP_projections{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        for patient in self.patient_data:
            # open the scan image in RGB, assign H and W (these should be 1536x1536 pixels - i.e. typical b-scan dims)
            img_path = patient["scan_path"]
            img = Image.open(img_path)
            arr = np.array(img, dtype=np.uint16)
            arr= _to_uint8(arr)
            img = Image.fromarray(arr)

            # make sure that the canvas is RGB, otherwise the circles won't be colour coded
            img = img.convert("RGB")

            # rotate image based on previously written left/right eye
            img = img.transpose(Image.Transpose.ROTATE_90)

            draw = ImageDraw.Draw(img)
            radius = 8

            # gather the calculated pixel and intensity info
            for stim in patient.get("computed_stim_values", []):
                px_x = stim["px_x"]
                px_y = stim["px_y"]
                intensity = stim["intensity"]

                # find the matching hex values to the intensity, as written in the enum
                # '<0' is a given value, so need to account for that with black
                colour_key = "<0" if intensity < 0 else str(intensity)

                # draw the circles for the intensity values, adding the corresponding colour map
                draw.ellipse([px_x - radius, px_y - radius, px_x + radius, px_y + radius],
                             fill=self.colour_map.get(colour_key),
                             outline="#FFFFFF",
                             width=1,
                             )
                draw.text((px_x, px_y), str(intensity), fill=_stim_text_hex(intensity), anchor="mm")

            p_id = patient.get("patient_id")
            exam_id = patient.get("ExamID")
            eye = patient.get("Eye")

            # add image identifier and save in output directory
            image_name = f"{p_id}_{exam_id}_{eye}"
            draw.text((1004, 20), image_name, fill="#FFFFFF", anchor="ra", font=self.font)
            save_path = os.path.join(save_dir, f"{image_name}_untransformed_MP_overlay.png")

            print(f"Saved: {save_path}")
            img.save(save_path)

    def overlay_thickness_maps_to_oct(self, registration_data):
        save_dir = os.path.join(self.output_dir, f"OCT_thickness_overlays_{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        for pair in registration_data:
            image_name = f"{pair['patient_id']}_{pair['exam_id']}_{pair['eye']}_{pair['timepoint']}"

            # one overlay per map: EZ-RPE map on the SLO and RPE-BM map on the SLO
            outputs = {'EZ_RPE': pair.get('ez_rpe_map_thickness_overlay'),'RPE_BM': pair.get('rpe_bm_map_thickness_overlay')}

            # keep the two map composites for overlay_stim_ellipses_to_oct
            pair['thickness_map_overlays'] = {label: img for label, img in outputs.items() if img is not None}

            for label, img in pair['thickness_map_overlays'].items():
                # copy so the stored overlay stays clean for the stimulus overlay step
                img = img.copy()
                ImageDraw.Draw(img).text((img.width - 20, 20), f"{image_name} - {label} 0-30um", fill="#FFFFFF", anchor="ra", font=self.font)

                save_path = os.path.join(save_dir, f"{image_name}_{label}_thickness_overlay.png")
                img.save(save_path)

                print(f"Saved: {save_path}")

    def overlay_transformed_stim_ellipses_to_oct(self, registered_pairs):
        save_dir = os.path.join(self.output_dir, f"Registered_MP_OCT_overlays_{self.timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        for pair in registered_pairs:
            transform = pair['transform_data']

            # transform the stim coordinates from the MP still into the SLO pixel frame
            stims = pair['stimuli']

            transf_stim_coords = transform.apply(np.array([[s['px_x'], s['px_y']] for s in stims]))

            # add the transformed x, y coordinates to the registration_data to be carried to the evaluations later
            for data, (t_x, t_y) in zip(stims, transf_stim_coords):
                data.update({'transf_px_x':t_x, 'transf_px_y':t_y})

            # draw on each thickness map composite, found in the pair>thickness_map_overlays
            for label, canvas in pair.get('thickness_map_overlays', {}).items():

                draw = ImageDraw.Draw(canvas)

                # draw the transformed stimuli, colour coded by intensity
                for stim, (px_x, px_y) in zip(stims, transf_stim_coords):
                    intensity = stim['intensity']
                    colour_key = "<0" if intensity < 0 else str(intensity)

                    draw.ellipse([px_x - 14, px_y - 14, px_x + 14, px_y + 14], fill=self.colour_map.get(colour_key), outline="#FFFFFF", width=3)
                    draw.text((px_x, px_y), str(intensity), fill=_stim_text_hex(intensity), anchor="mm", font_size=22)

                # image identifier and save
                image_name = f"{pair['patient_id']}_{pair['eye']}Eye_{label}_{pair['timepoint']}"
                draw.text((canvas.width - 20, 20), image_name, fill="#FFFFFF", anchor="ra", font=self.font)
                save_path = os.path.join(save_dir, f"{image_name}_MP_OCT_transformed.png")

                canvas.save(save_path)

                print(f"Saved: {save_path}")
