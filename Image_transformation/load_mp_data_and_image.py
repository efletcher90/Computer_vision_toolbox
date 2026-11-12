import xmltodict
import os

# --- File enums --- #
MP_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\MP_for_overlay'
# ------------------ #

# Create list of projection.xml microperimetry data
def get_mp_list():
    mp_list = []

    for dirpath, _, filenames in os.walk(MP_DIR):
        for xml_file in filenames:
            if xml_file.lower() == 'projection.xml':
                mp_list.append(os.path.join(dirpath, xml_file))

    return mp_list

def parse_mp_xml_data(xml_files):
    xml_dict = {}
    for x in xml_files:
        with open(x, "rb") as f:
            raw_bytes = f.read()
            # print(f"Parsing: {x} -> {raw_bytes[:60]}")
            data = xmltodict.parse(raw_bytes, force_list=("Stimulus",))
            stimuli = data["MitProjection"]["Stimuli"]["Stimulus"]

            for stimulus in stimuli:
                # need to add a zero to stimulus ids lower than 10, so that sorted works properly
                key = f"stimulus {int(stimulus['@id']):02d}"
                value = int(stimulus["@final_intensity"])
                xml_dict[key] = value

    sorted_xml_dict = dict(sorted(xml_dict.items()))
    return sorted_xml_dict

if __name__ == "__main__":
    mp_files = get_mp_list()
    parse_mp_xml_data(mp_files)