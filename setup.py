import inspect, json, pickle, os

import GPModels.SVGPModels as SVGP
import GPModels.ExactGPModels as EGP
import GPModels.MultiOutputExactGPModels as MOEGP

def create_json(module,file_name):
    members = inspect.getmembers(module)
    classes = [member for _, member in members if inspect.isclass(member) and member.__module__ == module.__name__]
    _classes = {}
    _json = {}
    for cls in classes:
        _classes[cls.__name__] = cls
        _json[cls.__name__] = False
    file = open(f'configs/{file_name}.json',"w")
    json.dump(_json,file,indent=4)
    file.close()

    return _classes

def setup():
    if not os.path.isdir("metadata"):
        os.mkdir("metadata")
    _svgp = create_json(SVGP,'svgp')
    file = open('metadata/svgp_dict',"wb")
    pickle.dump(_svgp,file)
    file.close()
    _exact_gp = create_json(EGP,'exact_gp')
    file = open('metadata/exact_gp_dict',"wb")
    pickle.dump(_exact_gp,file)
    file.close()
    _mo_exact_gp = create_json(MOEGP,'mo_exact_gp')
    file = open('metadata/mo_exact_gp_dict',"wb")
    pickle.dump(_mo_exact_gp,file)
    file.close()

if __name__ == "__main__":
    setup()