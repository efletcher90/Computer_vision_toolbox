import xmltodict
import os

# --- Enums -------- #
MP_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\MP_for_overlay'

METADATA_KEYS = ['ExamID', 'ExamBaselineID', 'Age', 'Eye', 'DateTime', 'fieldDegreesX', 'fieldDegreesY',
                 'baselinePRLX_pix', 'baselinePRLY_pix', 'baselineShiftX_pix', 'baselineShiftY_pix', 'fixationCenterX',
                 'fixationCenterY'
                 ]
# ------------------ #
class CreateMPDataDicts:
    def __init__(self, mp_dir, metadata_keys):
        self.mp_dir = mp_dir
        self.metadata_keys = metadata_keys

    # Create list of projection.xml microperimetry data
    def get_mp_list(self):
        mp_list = []
        # find all the scan folders within the root directory, then sort these and select only the latest scan to os.walk
        folders = sorted([os.path.join(self.mp_dir, name) for name in os.listdir(self.mp_dir) if os.path.isdir(os.path.join(self.mp_dir, name))])
        f = folders[-1]
        p_id = os.path.basename(f).split("_")[0]

        for dirpath, _, filenames in os.walk(f):
            for xml_file in filenames:
                if xml_file.lower() == 'projection.xml':
                    mp_list.append(os.path.join(dirpath, xml_file))

        return mp_list, p_id

    def parse_mp_xml_data(self, xml_files, p_id):
        all_xml_data = []

        for x in xml_files:
            xml_dict = {}
            stim_dict = {}

            with open(x, "rb") as f:
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
            xml_dict.update({'patient_id': p_id})
            m_data = mp_data["MitProjection"]["Header"]
            xml_dict.update({k: m_data[k] for k in self.metadata_keys if k in m_data})
            # the sorted stim data is added after the patient metadata so that the original order of the enum list is kept at the start of the dict
            xml_dict.update(sorted_stim_dict)

            all_xml_data.append(xml_dict)

        return all_xml_data

if __name__ == "__main__":
    xml_parser = CreateMPDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)
