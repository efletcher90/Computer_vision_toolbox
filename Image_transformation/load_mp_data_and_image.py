import xmltodict
import os
import math
from PIL import Image, ImageDraw, ImageFont

# --- Enums -------- #
# dir for mp and oct data
MP_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\MP_for_overlay'
OCT_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\OCT_masks_for_overlay'
# wanted keys to patient metadata stored in 'projection.xml' dict
METADATA_KEYS = ['ExamID', 'ExamBaselineID', 'Age', 'Eye', 'DateTime', 'fieldDegreesX', 'fieldDegreesY', 'examPRLX_pix', 'examPRLY_pix',
                 'baselinePRLX_pix', 'baselinePRLY_pix', 'baselineShiftX_pix', 'baselineShiftY_pix', 'fixationCenterX',
                 'fixationCenterY'
                 ]
# microperimetry db colour values, as shown in pdf files
DB_COLOUR_VALUES = {'<0':'#060608', '0':'#672B8D', '1':'#842577', '2':'#A31E5C', '3':'#C41E40', '4':'#E11E27', '5':'#EE2025', '6':'#EE2125', '7':'#EE2524', '8':'#EF2924', '9':'#EF3024', '10':'#EF3723', '11':'#F04022', '12':'#F04922', '13':'#F05122', '14':'#F15C23', '15':'#F16623', '16':'#F46F23', '17':'#F47722', '18':'#F68D1F', '19':'#FAA21C', '20':'#FEB714', '21':'#FDCC10', '22':'#F3EC0C', '23':'#E4E518', '24':'#D3DE25', '25':'#BCD531', '26':'#A3CD3A', '27':'#8AC53F','28':'#73BF43', '29':'#59BA46', '30':'#43B648', '31':'#31B349', '32':'#27B24A', '33':'#1DAE4A', '34':'#16A649', '35':'#119C48', '36':'#0E9245'}

# ------------------ #
class CreateMPDataDicts:
    def __init__(self, mp_dir, metadata_keys):
        self.mp_dir = mp_dir
        self.metadata_keys = metadata_keys

    # Create list of 'projection.xml' microperimetry datafiles
    def get_mp_list(self):
        mp_list = []

        # find all the scan folders within the root directory, then sort these and select only the latest scan to os.walk
        # (all previous exams are saved in the latest folder)
        folders = sorted([os.path.join(self.mp_dir, name) for name in os.listdir(self.mp_dir) if os.path.isdir(os.path.join(self.mp_dir, name))])
        f = folders[-1]
        # get the patient id from the file basename, just split at the first underscore. This is then assigned to the final dict
        p_id = os.path.basename(f).split("_")[0]

        for dirpath, _, filenames in os.walk(f):
            # gather lower case filenames in folder
            filenames = {files.lower(): files for files in filenames}

            if 'projection.xml' in filenames:
                xml_path = os.path.join(dirpath, filenames['projection.xml'])

                png_name = filenames.get('stillpicture.png')
                png_path = os.path.join(dirpath, png_name)

                mp_list.append({'xml_file': xml_path, 'scan_img': png_path})

        return mp_list, p_id

    def parse_mp_xml_data(self, xml_files, p_id):
        all_xml_data = []

        for x in xml_files:
            xml_path = x['xml_file']
            png_path = x['scan_img']

            xml_dict = {}
            stim_dict = {}

            with open(xml_path, "rb") as f:
                # read xml as raw bytes, convert to dict
                raw_bytes = f.read()
                mp_data = xmltodict.parse(raw_bytes, force_list=("Stimulus",))

                # parsing the .xml structure w/ xmltodict module. It creates stim_key=stimulus_id and stim_value=(final_intensity, ray, angle_deg)
                stimulus_data = mp_data["MitProjection"]["Stimuli"]["Stimulus"]
                for stimulus_id in stimulus_data:
                    # need to add a zero to stimulus ids lower than 10, so that sorted() works properly
                    stim_key = f"stimulus_{int(stimulus_id['@id']):02d}"
                    # values are a list of intensity, ray and angle_deg, in order.
                    # Cartesian positioning used to remap the coordinates from ray and angle_deg
                    stim_value = [int(stimulus_id["@final_intensity"]), float(stimulus_id["@ray"]), float(stimulus_id["@angle_deg"])]
                    stim_dict[stim_key] = stim_value

            sorted_stim_dict = dict(sorted(stim_dict.items()))

            # add patient metadata k:v pairs to xml_dict
            xml_dict.update({'patient_id': p_id, 'scan_path': png_path})
            m_data = mp_data["MitProjection"]["Header"]
            xml_dict.update({k: m_data[k] for k in self.metadata_keys if k in m_data})
            # the sorted stim data is added after the patient metadata so that the original order of the enum list is kept at the start of the dict
            xml_dict.update(sorted_stim_dict)

            all_xml_data.append(xml_dict)

        return all_xml_data
#
class CalculateStimCoordinates:
    def __init__(self, patient_data):
        self.patient_data = patient_data

    def load_patient_image(self):
        for patient in self.patient_data:
            # open the scan image in GS, assign H and W (these should be 1536x1536 pixels - i.e. typical b-scan dims)
            img_path = patient["scan_path"]
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

            # using the fieldDegreesX/Y values from the xml, scale the image
            scaled_x = W / float(patient['fieldDegreesX'])
            scaled_y = H / float(patient['fieldDegreesY'])

            # using the baselinePRLX/Y_pix and baselineShiftX/Y, find the centre point of the image relative to the baseline image.
            # For BL exams images, only the exam examPRLX_pix is used hree
            exam_bl_id = patient['ExamBaselineID']
            if exam_bl_id != "-1" or "0":
                centre_x = W / 2 + patient["baselinePRLX_pix"] + patient["baselineShiftX_pix"]
                centre_y = W / 2 + patient["baselinePRLY_pix"] + patient["baselineShiftY_pix"]
            else:
                centre_x = W / 2 + patient["examPRLX_pix"]
                centre_y = W / 2 + patient["examPRLY_pix"]


    # def calculate_coordinate_pixels(self):
    #
    # def map_stim_ellipses(self ):

# class OverlayValuesToFundus:
#     def __init__(self):

if __name__ == "__main__":
    xml_parser = CreateMPDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)

    stim_coords = CalculateStimCoordinates(patient_data_dicts)

