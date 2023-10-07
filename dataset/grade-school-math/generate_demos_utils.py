import json
import random

def generate_prompts_from_json(input_file, output_file, num_samples):
    # Load the data from the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    randomlist = random.sample(range(0, len(data["x"])), num_samples)
    
    # Lists to store the generated prompts
    x=[]
    y=[]
    z=[]
    z_uninformative=[]
    for i in randomlist:
        x.append(data["x"][i])
        y.append(data["y"][i])
        z.append(data["z"][i])
        z_uninformative.append(data["z_uninformative"][i])
    
    
    # Create the output data structure
    output_data = {
        "x": x,
        "y": y,
        "z": z,
        "z_uninformative": z_uninformative
    }
    
    # Save to the output JSON file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    return output_data

