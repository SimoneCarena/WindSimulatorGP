import inspect, json, pickle, os, re

import GPModels.SVGPModel as SVGP
import GPModels.ExactGPModel as EGP
import GPModels.MultiOutputExactGPModel as MOEGP

def __create_json(module,file_name):
    members = inspect.getmembers(module)
    classes = [member for _, member in members if inspect.isclass(member) and member.__module__ == module.__name__]
    _classes = {}
    _json = {}
    module_name = re.sub('.*\\.','',module.__name__)
    for cls in classes:
        _classes[re.sub(module_name,'',cls.__name__)] = cls
        _json[re.sub(module_name,'',cls.__name__)] = False
    file = open(f'configs/{file_name}.json',"w")
    json.dump(_json,file,indent=4)
    file.close()

    return _classes

def __setup_json():
    if not os.path.isdir(".metadata"):
        os.mkdir(".metadata")
    _svgp = __create_json(SVGP,'svgp')
    file = open('.metadata/svgp_dict',"wb")
    pickle.dump(_svgp,file)
    file.close()
    _exact_gp = __create_json(EGP,'exact_gp')
    file = open('.metadata/exact_gp_dict',"wb")
    pickle.dump(_exact_gp,file)
    file.close()
    _mo_exact_gp = __create_json(MOEGP,'mo_exact_gp')
    file = open('.metadata/mo_exact_gp_dict',"wb")
    pickle.dump(_mo_exact_gp,file)
    file.close()

def __setup():
    __setup_json()

if __name__ == "__main__":
    __setup()