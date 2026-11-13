import xmltodict
import os

# --- File enums --- #
MP_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\MP_for_overlay'
# ------------------ #

# Create list of projection.xml microperimetry data
def get_mp_list():
    mp_list = []
    # find all the scan folders within the root directory, then sort these and select only the latest scan to os.walk
    folders = sorted([os.path.join(MP_DIR, name) for name in os.listdir(MP_DIR) if os.path.isdir(os.path.join(MP_DIR, name))])
    f = folders[-1]
    p_id = os.path.basename(f).split("_")[0]

    for dirpath, _, filenames in os.walk(f):
        for xml_file in filenames:
            if xml_file.lower() == 'projection.xml':
                mp_list.append(os.path.join(dirpath, xml_file))

    return mp_list, p_id

def parse_mp_xml_data(xml_files, p_id):
    xml_dict = {}
    patient_key = f"patient_id"
    xml_dict[patient_key] = p_id

    for x in xml_files:
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
                xml_dict[stim_key] = stim_value

            # stimulus_data = mp_data["MitProjection"]["Stimuli"]["Stimulus"]

    sorted_xml_dict = dict(sorted(xml_dict.items()))
    return sorted_xml_dict

if __name__ == "__main__":
    mp_files, patient_id = get_mp_list()
    parse_mp_xml_data(mp_files, patient_id)