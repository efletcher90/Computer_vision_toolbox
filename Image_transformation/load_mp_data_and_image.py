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

    for dirpath, _, filenames in os.walk(f):
        for xml_file in filenames:
            if xml_file.lower() == 'projection.xml':
                mp_list.append(os.path.join(dirpath, xml_file))

    return mp_list

def parse_mp_xml_data(xml_files):
    xml_dict = {}
    for x in xml_files:
        with open(x, "rb") as f:
            raw_bytes = f.read()
            stim_data = xmltodict.parse(raw_bytes, force_list=("Stimulus",))
            # parsing the .xml structure w/ xmltodict module. It creates key=stimulus_id and value=final_intensity
            stimuli = stim_data["MitProjection"]["Stimuli"]["Stimulus"]

            for stimulus in stimuli:
                # need to add a zero to stimulus ids lower than 10, so that sorted() works properly
                key = f"stimulus {int(stimulus['@id']):02d}"
                # values are a list of intensity, ray and angle_deg, in order.
                # Cartesian positioning used to remap the coordinates from ray and angle_deg
                value = [int(stimulus["@final_intensity"]), float(stimulus["@ray"]), float(stimulus["@angle_deg"])]
                xml_dict[key] = value

    sorted_xml_dict = dict(sorted(xml_dict.items()))
    return sorted_xml_dict

if __name__ == "__main__":
    mp_files = get_mp_list()
    parse_mp_xml_data(mp_files)