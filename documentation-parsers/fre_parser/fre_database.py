#!/usr/bin/env python3

import copy

from parser import ModuleDocument, CommandDocument

class FreDatabase:

    TOOLS_DICT: dict[str, dict[str, ModuleDocument | None]] = {
        "make": {
            "fre.make.create_checkout_script": None,
            "fre.make.create_compile_script": None,
            "fre.make.create_docker_script": None,
            "fre.make.create_makefile_script": None,
            "fre.make.make_helpers": None,
            "fre.make.run_fremake_script": None
        },
        "yaml": {
            "fre.yamltools.abstract_classes": None,
            "fre.yamltools.combine_yamls_script": None,
            "fre.yamltools.constructors": None,
            "fre.yamltools.helpers": None,
        },
        "app": {
            "fre.app.generate_time_averages.cdoTimeAverager": None,
            "fre.app.generate_time_averages.combine": None,
            "fre.app.generate_time_averages.frenctoolsTimeAverager": None,
            "fre.app.generate_time_averages.frepytoolsTimeAverager": None,
            "fre.app.generate_time_averages.generate_time_averages": None,
            "fre.app.generate_time_averages.timeAverager": None,
            "fre.app.generate_time_averages.wrapper": None,
            "fre.app.mask_atmos_plevel.mask_atmos_plevel": None,
            "fre.app.regrid_xy.regrid_xy": None,
            "fre.app.remap_pp_components.remap_pp_components": None,
        },
        "list": {
            "fre.list_.list_experiments_script": None,
            "fre.list_.list_platforms_script": None,
            "fre.list_.list_pp_components_script": None,
        },
        "pp": {
            "fre.pp.checkout_script": None,
            "fre.pp.configure_script_yaml": None,
            "fre.pp.histval_script": None,
            "fre.pp.install_script": None,
            "fre.pp.nccheck_script": None,
            "fre.pp.ppval_script": None,
            "fre.pp.rename_split_script": None,
            "fre.pp.run_script": None,
            "fre.pp.split_netcdf_script": None,
            "fre.pp.status_script": None,
            "fre.pp.trigger_script": None,
            "fre.pp.validate_script": None,
            "fre.pp.wrapper_script": None
        },
        "run": {
            "fre.run.frerunexample": None
        },
    }

    COMMANDS_DICT: dict[str, CommandDocument | None] = {
        "fre.make.fremake": None,
        "fre.yamltools.freyamltools": None,
        "fre.app.freapp": None,
        "fre.catalog.frecatalog": None,
        "fre.list_.frelist": None,
        "fre.pp.frepp": None,
        "fre.run.frerun": None
    }

    def __init__(self, module_name=None):
        """Constructor"""
        module_found = False
        if module_name:
            for mod in self.TOOLS_DICT.keys():
                if module_name == mod:
                    self.tools_dict = {}
                    self.commands_dict = {}
                    self.tools_dict[mod] = self.TOOLS_DICT[mod]
                    self.commands_dict[f"fre.{mod}.fre{mod}"] = self.COMMANDS_DICT[f"fre.{mod}.fre{mod}"]\
                                                                if mod != "list"\
                                                                else self.COMMAND_DICT[f"fre.{mod}_.fre{mod}"]
                    module_found = True
                    break

        if not module_found or not module_name:
            self.tools_dict = copy.deepcopy(self.TOOLS_DICT)
            self.commands_dict = copy.deepcopy(self.COMMANDS_DICT)

    def summarize(self):
        """Summarizes the docstrings in the modules and commands"""
        
        for tool, tool_dict in self.tools_dict.items():
            for modulename in tool_dict:
                try:
                    moduledocument = ModuleDocument(modulename)
                    moduledocument.summarize()
                    self.tools_dict[tool][modulename] = moduledocument
                except Exception as exc:
                    print(f"Skipping module {modulename}: {exc}")
        
        for command in self.commands_dict:
            try:
                commanddocument = CommandDocument(command)
                commanddocument.summarize()
                self.commands_dict[command] = commanddocument
            except Exception as exc:
                print(f"Skipping command group {command}: {exc}")

    def to_chromadb(self):
        """Converts the summarized data into document_list, metadata_list, and id_list for ChromaDB"""
        
        document_list, metadata_list, id_list = [], [], []

        for tool, tool_dict in self.tools_dict.items():
            for modulename, moduledocument in tool_dict.items():
                if moduledocument is None:
                    continue
                for function in moduledocument.function_summaries:
                    document_list.append(moduledocument.module_overview + moduledocument.function_summaries[function])
                    metadata_list.append(moduledocument.function_metadata[function])
                    id_list.append(f"{modulename}.{function}")
        
        for command, commanddocument in self.commands_dict.items():
            if commanddocument is None:
                continue
            for subcommand in commanddocument.subcommand_summaries:
                document_list.append(commanddocument.module_overview + commanddocument.subcommand_summaries[subcommand])
                metadata_list.append(commanddocument.subcommand_metadata[subcommand])
                id_list.append(f"{command}.{subcommand}")         
        
        return document_list, metadata_list, id_list

if __name__ == '__main__':

    fre_db = FreDatabase()
    fre_db.summarize()
    document_list, metadata_list, id_list = fre_db.to_chromadb()
