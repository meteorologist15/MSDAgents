import importlib
import inspect
import re
from typing import TypedDict

import click


class MetadataEntry(TypedDict):
    name: str
    module: str
    package: str


class DocumentUtility():

    """
    Base class for documents. Contains utility functions for parsing docstrings
    """

    def __init__(self):
        """Constructor"""
        pass

    def _check(self, docstring):
        """
        Return empty string if docstring is None
        """
        if docstring is not None:
            docstring_stripped = docstring.strip()
            if docstring_stripped:
                return docstring_stripped
        return ""

    def _clean(self, string_in):
        """
        Remove empty lines and excessive whitespace from a string.
        """
        for escape_char in ("\n", "\t", "\r", "\b", "\f", "\v", "\0"):
            string_in = string_in.replace(escape_char, " ")
        string_in = re.sub(r' +', ' ', string_in)
        return string_in.strip()


class ModuleDocument(DocumentUtility):

    """
    Class to parse and summarize docstrings in a module
    Example use case:
    self.packagename = "fre"
    self.modulename = "fre.make.create_checkout_script"
    self.docstring_dict = {
        "baremetal_checkout_write": {
        "overview": "baremetal_checkout_write is called by checkout_create..."
        "params": [
            "model_yaml is a 'freyaml' class object containing.. .model_yaml is of type yamlfre.freyaml",
            "src_dir is the absolute directory path to git clone the source code. src_dir is of type str."
        ],
        "raises": [
            "ValueError is raised when platform does not exist in platforms.yaml."
        ],
        "notes": "This is the docstring starting with .. note::"
        }
    }
    self.module_overview = "top level docstring for the module"
    self.function_summaries = {"baremetal_checkout_write": 
        "baremetal_checkout_write is called by checkout_create...
            The function has the following parameters: model_yaml is a 'freyaml' class object containing.. .
            model_yaml is of type yamlfre.freyaml. 
            src_dir is the absolute directory path to git clone the source code. src_dir is of type str.
            The following errors can be raised: ValueError is raised when platform does not exist in platforms.yaml. 
            Note: This is the docstring starting with .. note::"
    }
    self.function_metadata = {
        "baremetal_checkout_write": {
            "name": "baremetal_checkout_write",
            "module": "fre.make.create_checkout_script",
            "package": "fre"
        }
    }
    """

    def __init__(self, modulename: str, packagename: str = "fre"):
        """Constructor"""
        super().__init__()
        self.packagename = packagename
        self.modulename = modulename
        
        self.module_overview = ""
        self.function_metadata: dict[str, MetadataEntry] = {}
        self.function_summaries = {}
        self.docstring_dict = {}
        
        self.mod = importlib.import_module(self.modulename)

    def summarize(self):
        """
        Parses the module file and saves docstrings into this object.
        """

        #set top level module documentation.  if None, set to empty string
        self.module_overview = self._check(inspect.getdoc(self.mod))
                
        #for each function in module
        for name, func in inspect.getmembers(self.mod, inspect.isfunction):
            
            #if function is not a module function (i.e. is a built-in function), skip
            if inspect.getmodule(func) is not self.mod: 
                continue
            
            #split docstring into overview, params, raises, and notes
            splitted_docstring = self.split_docstring(
                inspect.getdoc(func)
            )

            #parse docstrings into lists of sentences
            docstring_dict= {
                "overview": self._clean(splitted_docstring["overview"]),
                "params": self.parse_params(splitted_docstring["params"]),
                "raises": self.parse_raises(splitted_docstring["raises"]),
                "notes": self.parse_notes(splitted_docstring["notes"])
            }
            
            #store
            self.docstring_dict[name] = splitted_docstring
            
            #convert doc_dict to paragraphs
            self.function_summaries[name] = self.summarize_docstring_dict(docstring_dict)
            
            #save metadata 
            self.function_metadata[name] = {
                "name": name,
                "module": self.modulename,
                "package": self.packagename
            }

    def summarize_docstring_dict(self, docstring_dict):
        """
        Converts the sentences in self.functions into a summary.
        """
        
        summary = docstring_dict["overview"]
        
        if docstring_dict["params"]:
            summary += " The function has the following parameters:\n"
            for param in docstring_dict["params"]:
                summary += f"- {param}\n"
        
        if docstring_dict["raises"]:
            summary += " The function can raise the following exceptions:\n"
            for raise_ in docstring_dict["raises"]:
                summary += f"- {raise_}\n"
        
        if docstring_dict["notes"]:
            summary += " Note: " + docstring_dict["notes"]
            
        return summary

    def split_docstring(self, docstring_in):
        """
        Splits a function docstring into overview, params, raises, and notes
        Returns empty strings for empty fields
        """
        
        docstring = self._check(docstring_in)
        
        notes = raises = params = ""
        
        if docstring:
            if ".. note::" in docstring:
                docstring, notes = docstring.split(".. note::", 1)
            if ":raises" in docstring:
                docstring, raises = docstring.split(":raises", 1)
            if ":param" in docstring:
                docstring, params = docstring.split(":param", 1)

        return {
            "overview": self._check(docstring),
            "raises": self._check(raises),
            "notes": self._check(notes),
            "params": self._check(params)
        }

    def parse_params(self, params_docstring: str) -> list[str]:
        """
        Parses docstrings such as
        ```
        :param src_dir: is the absolute directory path to git clone the source code
        :type src_dir: str
        ```
        and saves the content as the two sentences below:
        ```
        src_dir is the absolute directory path to git clone th source code.
        src_dir is of type str
        ```
        """
        
        if not self._check(params_docstring): return []
        
        params = []
        for param_and_type_string in params_docstring.split(":param"):
            try:
                paramstuff, typestuff = param_and_type_string.split(":type", 1)
            except Exception as exc:
                raise RuntimeError(f"Could not parse {param_and_type_string} into param and type") from exc
            paramname, paramstring = paramstuff.split(":", 1)
            type_ = typestuff.split(":", 1)[1]
            
            paramname = self._check(paramname)
            params.append(
                f"{paramname} {self._clean(paramstring)}.  "
                f"{paramname} is of type {self._clean(type_)}.  "
            )

        return params
    
    def parse_raises(self, raises_docstring) -> list[str]:
        """
        Parses docstrings such as ':raises ValueError: Error if platform does not exist in platforms.yaml'
        and saves the content as 'ValueError is raised if platform does not exist in platforms.yaml.'
        """

        if not self._check(raises_docstring): return []
        
        raises = []
        for a_docstring in raises_docstring.split(":raises"):
            raise_type, raise_condition = a_docstring.split(":", 1)
            raises.append(
                f"{self._clean(raise_type)} is raised when {self._clean(raise_condition)}."
            )

        return raises

    def parse_notes(self, notes_docstring) -> str:
        """
        Parses docstrings such as '.. note:: This is some note'        
        and saves the content as 'This is some note.'
        """    
        if not self._check(notes_docstring): return ""
        return self._clean(notes_docstring)

    def __repr__(self):
        """Pretty print for instance of this class"""
        functions = []
        for name, report in self.function_summaries.items():
            functions.append(f"* {name}:\n{report}\n")

        functions_text = "\n".join(functions)

        return (
            "__________________________________\n"
            f"MODULE:\n{self.modulename}\n\n"
            f"Overview:\n{self.module_overview}\n\n"
            f"Functions:\n{functions_text}"
            "__________________________________"
        )


class CommandDocument(DocumentUtility):    

    """
    Class to convert docstrings in a module to a report format for Click commands.
    Example use case:
    self.packagename = "fre"
    self.modulename = "fre.make.fremake"
    self.docstring_dict = {
        "all": {
            "overview": "all is the main command that calls all other commands in the group.",
            "help": "output from fre make all --help"
        }
    }
    self.module_overview = "top level docstring for the module"
    self.subcommands = {"all":
        "all is the main command that calls all other commands in the group.
            The help information for this command is as follows: ..."
    }
    self.metadata = {
        "all": {
            "name": "all",
            "module": "fre.make.fremake",
            "package": "fre"
        }
    }
    """

    def __init__(self, modulename: str, packagename: str = "fre"):
        """Constructor"""

        super().__init__()
        self.modulename = modulename
        self.packagename = packagename        
        self.module_overview = None
        self.docstring_dict = {}
        self.subcommand_summaries = {}
        self.subcommand_metadata = {}

    def summarize(self):
        """
        Parses the module file and saves Click subcommand docstrings
        """
        mod = importlib.import_module(self.modulename)

        #set module overview
        self.module_overview = self._check(inspect.getdoc(mod))

        for name in dir(mod):            
            obj = getattr(mod, name)
            # if obj is an a Click command
            if isinstance(obj, click.Command): #and obj.name == self.groupcommand:
            
                docstring = inspect.getdoc(obj)
                help_output = obj.get_help(click.Context(obj))

                #save
                command_dict = {
                    "overview": self._clean(self._check(docstring)),
                    "help": self._check(help_output)
                }

                #store
                self.docstring_dict[name] = command_dict

                #convert command_dict to paragraph
                self.subcommand_summaries[name] = self.summarize_docstring_dict(command_dict)

                #save metadata
                self.subcommand_metadata[name] = {
                    "name": name,
                    "module": self.modulename,
                    "package": self.packagename
                }

    def summarize_docstring_dict(self, command_dict):
        """
        Summarizes the command docstring and help information into a paragraph.
        """
        report = (
            f"{command_dict['overview']} "
            f"The help information for this command is as follows:\n {command_dict['help']}"
        )
        return report

    def __repr__(self):
        """Pretty print for instance of this class"""
        commands = []
        for name, report in self.subcommand_summaries.items():
            commands.append(f"* {name}:\n{report}\n")

        commands_text = "\n".join(commands)

        return (
            "__________________________________\n"
            f"MODULE:\n{self.modulename}\n\n"
            f"Overview:\n{self.module_overview}\n\n"
            f"Commands:\n{commands_text}"
            "__________________________________"
        )
