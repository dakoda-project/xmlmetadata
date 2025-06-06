import os
import re
import sys
import glob
import copy
import pprint as pp
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import pandas as pd
from lxml import etree
import toml
import nltk
from cassis import *
from dkd_meta import dakoda_metadata_scheme_2025_06_04 as meta

# import xsdata.formats.dataclass.serializer.XmlSerializer as XmlSerializer
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers import JsonSerializer


jserializer = JsonSerializer()

from xsdata.formats.dataclass.serializers.config import SerializerConfig


serializer = XmlSerializer(
    config=SerializerConfig(
        xml_declaration=False,
        pretty_print=True,
        ignore_default_attributes=False,
        schema_location="/mnt/biggie/dakoda_metadaten/xml_json_schema_2025/dakoda_metadata_scheme_2025-05-26.xsd",
    )
)


def setup_logger(logger_name, logfile):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter(
    #   "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    # formatter = logging.Formatter("%(asctime)s %(levelname)s — %(message)s",  "%y-%m-%d %H:%M:%S")
    formatter = logging.Formatter("%(asctime)s %(levelname)s — %(message)s", "%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# setup the logger as below with mylogger being the name of the #logger and myloggerfile.log being the name of the logfile
logfilename = re.sub("py$", "log.tsv", sys.argv[0])
if os.path.exists(logfilename):
    os.remove(logfilename)

logger = setup_logger("mylogger", logfilename)
logger.info("My Logger has been initialized")


import argparse


ABOUT = """
What we're all about: generate xml and  json metadata files by pulling infomration from various sources.

 """
parser = argparse.ArgumentParser(description=ABOUT)


default_toml_config_file = "wtld_metadata_conversion.toml"
parser.add_argument(
    "-t",
    action="store",
    dest="toml_config_file",
    default=default_toml_config_file,
    help=f"path to toml configuration file (default: {default_toml_config_file})",
)


default_maxfiles = 5  # 50
parser.add_argument(
    "-m",
    action="store",
    dest="maxfiles",
    default=default_maxfiles,
    type=int,
    help=f"number of files to process at most (default: {default_maxfiles})",
)


args = parser.parse_args()

toml_config_file = args.toml_config_file

with open(f"/mnt/biggie/jkrcode/paula2cas/corpconfig/{toml_config_file}", "r") as f:
    config = toml.load(f)

args = parser.parse_args()


LANG_VALS_MAP = {"f": "fra", "e": "eng", "d_e": "deu,eng", "d": "deu"}

GENDER_MAP = {"f": "female", "m": "male", "n": "non-binary"}


def lang_abbr_mapping(l1a_str):
    if l1a_str == "NA":
        return "notAvailable"
    else:
        return l1a_str


def get_lan_info_for_iso_code(
    lan_string: str, lang_info_df: pd.DataFrame
) -> Tuple[str, str]:
    """
    Retrieves language names in English and German for a given ISO language code.

    """

    logger.info(f"Looking up language info for ISO code {lan_string}")

    lang_df = lang_info_df[lang_info_df["speaker_l1_1_iso639_3_code"] == lan_string]
    if len(lang_df) == 1:
        lang_row = lang_df.iloc[0]
        lan_name_en = lang_row["speaker_l1_1_iso639_3_en"]
        lan_name_de = lang_row["speaker_l1_1_iso639_3_de"]
    else:
        logger.error(f"could not retrieve info for language with code {lan_string}")
        lan_name_en = meta.NaString.NOT_AVAILABLE
        lan_name_de = meta.NaString.NOT_AVAILABLE
    return (lan_name_en, lan_name_de)


#  in_DAKODA    L1_DKD  L2_DKD  Orig_Wert_deutscheBezeichnung   Orig_Wert_englischeBezeichnung
#  Orig_code_ISO    Orig_code_anderer   iso639-3_de iso639-3_en     iso639-3_code   dominantWordOrder


def lookup_word_order(df: pd.DataFrame, code: str) -> str:
    logger.info(f"Looking up word order for ISO code {code}")
    match_df = df[df["speaker_l1_1_iso639_3_code"] == code]
    if len(match_df) == 0:
        return "notAvailable"
    else:
        assert len(match_df) == 1
        return match_df.iloc[0]["speaker_l1_1_dominantWordOrder"]


REF1 = """🔥 Karges, K., Studer, T., & Wiedenkeller, E. (2019). On the way to a new multilingual learner corpus of foreign language learning in school: Observations about task variation. In A. Abel, A. Glaznieks, V. Lyding, & L. Nicolas (Eds.), Widening the Scope of Learner Corpus Research. Selected papers from the fourth Learner Corpus Research Conference (pp. 137–165). Presses universitaires de Louvain."""


def look_up_swiko_task_description(task_id, df):
    """get info from teh task dataframe"""
    logger.info(f"Looking up swiko task description for task id {task_id}")
    return df[df["taskid"] == task_id]["description"].iloc[0]



DKD_CORPUS_CONTRIBS = [
    "Jamila Bläsing",
    "Luise Böttcher",
    "Shanny Druker",
    "Lisa Lenort",
    "Annette Portmann",
    "Christine Renker",
    "Josef Ruppenhofer",
    "Matthias Schwendemann",
    "Iulia Sucutardean",
    "Katrin Wisniewski",
    "Torsten Zesch",
]


def map_to_boolean(input_string):
    """
    Converts various string representations of boolean values to actual boolean values.

    This function handles different text representations of boolean values in multiple
    languages and formats, as well as special cases for missing values.

    Parameters:
            input_string (str): The string to convert to a boolean value. Can be various
                                               representations like 'true', 'yes', 'ja', '1' for True or
                                               'false', 'no', 'nein', '0' for False. Can also be missing
                                               value indicators like NaN, 'NA', 'nan', 'none'.

    Returns:
            bool or meta.NaString.NOT_AVAILABLE or None:
                    - True if the input represents a truthy value
                    - False if the input represents a falsy value
                    - meta.NaString.NOT_AVAILABLE if the input represents a missing value
                    - None if the input doesn't match any known pattern
    """
    if input_string.lower() in ["true", "yes", "ja", "1"]:
        return True
    elif input_string.lower() in ["false", "no", "nein", "0"]:
        return False
    elif pd.isna(input_string):
        return meta.NaString.NOT_AVAILABLE
    elif input_string.lower() in ["na", "nan", "none"]:
        return meta.NaString.NOT_AVAILABLE
    elif int(input_string) == 1:
        return True
    elif int(input_string) == 0:
        return False
    else:
        return None


def convert_cell_string_to_list(cell_string):
    """
    Converts a semicolon-separated string from a spreadsheet cell into a list of strings.

    This function splits the input string by semicolons, trims whitespace from each component,
    replaces non-breaking spaces with regular spaces, and converts special markers ('#')
    to standardized 'NOT_AVAILABLE' values.

    Parameters:
            cell_string (str): The input string from a spreadsheet cell, with components separated by semicolons.

    Returns:
            list: A list of cleaned string values, with special markers converted to standardized values.
    """
    outlist = []
    comps_list = re.split("\s*;\s*", cell_string)
    for rawcomp in comps_list:
        raw_cl = rawcomp.strip()
        raw_cl = raw_cl.replace("\xa0", " ")
        if raw_cl == "#":
            raw_cl = meta.NaString.NOT_AVAILABLE
        outlist.append(raw_cl)
    return outlist


def count_l1s(row, collist):
    """
    This function counts the number of non-NA values in a specified list of columns for a given row.

    Parameters:
    row (pandas Series): The row of data to be analyzed.
    collist (list of str): The list of column names to be checked for non-NA values.

    Returns:
    int: The count of non-NA values in the specified columns for the given row.
    """
    lcount = 0
    for col in collist:
        if not pd.isna(row[col]):
            # in theory we maybe should validate the cell value, too!
            lcount += 1
    return lcount


def main():
    # CSECTION: load info files into dataframes
    # docids, task ids, corpus metadata, language info, learner info, transciber info
    docidfile = config["corpus"]["docid_file"]
    with open(docidfile, "r", encoding="utf-8") as inc:
        docid_lines = inc.readlines()

    # NB: here if we want to, we can do filtering here for subsets of Ids
    # docid_lines = [x for x in docid_lines if "ri" in x.lower()]
    print(f"{len(docid_lines)} docids")

    # header: taskid, description
    # the description comes from a pdf in the corpus folder
    task_df = pd.read_csv(
        config["corpus"]["task_description_table"], sep="\t", quoting=3
    )

    # 62 rows for different corpora as of Mo 02 Jun 2025 ( KW22 )
    corp_meta_df = pd.read_excel(
        config["corpus"]["corpus_metadata_table"], engine="openpyxl", header=0
    )
    print(corp_meta_df.head())

    # this df  has info on names and word order, among others
    lang_info_df = pd.read_csv(
        config["corpus"]["lang_info_lookup"], sep="\t", quoting=3
    )
    print(f"{len(lang_info_df)} languages")
    print(lang_info_df.head())

    # info on the students/learners
    learner_metadata_table = config["corpus"]["learner_metadata_table"]
    # PID   schoolyear  age text_learner_ageProduction  gender  learner_gender  l1a l1b l1c subcorpus   learner_school_language yearcollected   canton  class   learner_durationOfInstruction   learner_ageofOnset
    learner_df = pd.read_csv(learner_metadata_table, sep="\t", quoting=3)
    logger.debug(learner_df.head())

    # info on transcibers of original corpus, potentially for use as annotators ?!
    transcriber_df = pd.read_csv(
        config["corpus"]["transcriber_table"],
        sep="\t",
        quoting=3,
        names=["file", "trans_sigle"],
    )
    doc2trans_map = {}
    for ix, row in transcriber_df.iterrows():
        name_bits = re.split("\_", row["file"])
        shortname = "_".join(name_bits[0:3])
        if shortname not in doc2trans_map:
            doc2trans_map[shortname] = []
        else:
            logger.error(
                "there are paper and computer versions of the same single task instance!"
            )

        doc2trans_map[shortname] = row["trans_sigle"]

    logger.debug(f"doc2trans_map: {len(doc2trans_map)} entries\n {doc2trans_map}")

    corpus_name = config["corpus"]["name"]

    MAX_DOCS_TO_PROC = args.maxfiles

    # fill learner map
    # two letters at zero-based pos 1: first letter (lower case)  is language of schooling, second (capital letter) is language of text
    # SWI26_dD_Ba253_p
    learner_2_taskIds_map = {}
    for ix, docid in enumerate(docid_lines):
        comps = re.split("_", docid)

        task_id = comps[0]

        school_and_text_langs = comps[1]
        src_learner_id = comps[2]
        p_or_c = comps[3]

        if src_learner_id not in learner_2_taskIds_map:
            learner_2_taskIds_map[src_learner_id] = []
        learner_2_taskIds_map[src_learner_id].append(task_id)

    outdir = config["corpus"]["metadata_output_dir"]

    # CSECTION: collect  info related to the corpus as a whole
    # we don't want to look these things up again for each document
    curr_corp_name = corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name].iloc[
        0
    ]["corpus_name"]

    curr_corp_author_string = (
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name]
        .iloc[0]["corpus_author"]
        .strip()
    )
    curr_corp_authors = re.split("\s*;\s*", curr_corp_author_string)

    curr_corp_availability = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_availability"]
    if curr_corp_availability == "#":
        curr_corp_availability = meta.NaString.NOT_AVAILABLE

    curr_corp_cite_as = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_cite_as"]
    if curr_corp_cite_as == "#":
        curr_corp_cite_as = meta.NaString.NOT_AVAILABLE

    corpus_contact_mail_string = (
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name]
        .iloc[0]["corpus_contact_mail"]
        .strip()
    )
    curr_corp_email_addresses = re.split("\s*;\s*", corpus_contact_mail_string)

    curr_corp_contribs_string = (
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name]
        .iloc[0]["corpus_contributors_orig"]
        .strip()
    )
    curr_corp_contribs_list = re.split("\s*;\s*", curr_corp_contribs_string)

    curr_corp_date_of_pub_orig = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_date_of_publication_orig"]

    # Q: shouldn't we give the SWIKO Manual as documentation?
    curr_corp_documentation = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_documentation"]
    if curr_corp_documentation == "#":
        curr_corp_documentation = meta.NaString.NOT_AVAILABLE

    curr_corp_licence = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_licence"]
    if curr_corp_licence == "#":
        curr_corp_licence = meta.NaString.NOT_AVAILABLE

    curr_corp_licence_url = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_licence_url"]
    if curr_corp_licence_url == "#":
        curr_corp_licence_url = meta.NaString.NOT_AVAILABLE

    curr_corp_licenceFulltext = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_licenceFulltext"]
    if curr_corp_licenceFulltext == "#":
        curr_corp_licenceFulltext = meta.NaString.NOT_AVAILABLE

    curr_corp_ref_article = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_ref_article"]
    if curr_corp_ref_article == "#":
        curr_corp_ref_article = meta.NaString.NOT_AVAILABLE

    curr_corp_pid_dkd = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_pid_dkd"]
    if curr_corp_pid_dkd == "#":
        curr_corp_pid_dkd = meta.NaString.NOT_AVAILABLE

    curr_corp_pid_orig = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_pid_orig"]
    if curr_corp_pid_orig == "#":
        curr_corp_pid_orig = meta.NaString.NOT_AVAILABLE

    curr_corp_other_versions_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_other_versions"]
    curr_corp_other_versions_list = re.split("\s*;\s*", curr_corp_other_versions_string)

    curr_corps_referencesOther_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_referencesOther"]
    curr_corps_referencesOther_list = []
    raw_curr_corps_referencesOther_list = re.split(
        "\s*;\s*", curr_corps_referencesOther_string
    )
    for raw_ref in raw_curr_corps_referencesOther_list:
        raw_cl = raw_ref.strip()
        if raw_cl == "#":
            raw_cl = meta.NaString.NOT_AVAILABLE
        curr_corps_referencesOther_list.append(raw_cl)

    curr_corp_researchPapers_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_researchPapers"]
    curr_corp_researchPapers_list = convert_cell_string_to_list(
        curr_corp_researchPapers_string
    )

    curr_corp_URL_download_str = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_URL_download"]
    curr_corp_URL_download_list = convert_cell_string_to_list(
        curr_corp_URL_download_str
    )

    curr_corp_designType = map_to_boolean(
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name].iloc[0][
            "corpus_longitudinal"
        ]
    )

    curr_corp_corpusGroup = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_group"]

    curr_corp_comparable_data_included = map_to_boolean(
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name].iloc[0][
            "corpus_comparable_data_included"
        ]
    )

    # corpus_L1_language
    curr_corp_design_l1_language_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_L1_language"]
    if curr_corp_design_l1_language_val == "#":
        curr_corp_design_l1_language_val = meta.NaString.NOT_AVAILABLE

    # corpus_L1_type (mono, multi,NA)
    curr_corp_design_l1_type_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_L1_type"]

    curr_corp_design_target_language_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_target_language"]
    curr_corp_design_target_language_list = convert_cell_string_to_list(
        curr_corp_design_target_language_string
    )

    # corpus_target_language_type: list of strings
    curr_corp_design_target_language_type_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_target_language_type"]
    curr_corp_design_target_language_type_list = convert_cell_string_to_list(
        curr_corp_design_target_language_type_string
    )

    curr_corp_design_subcorpus_target_language_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_subcorpus_target_language"]
    curr_corp_design_subcorpus_target_language_list = convert_cell_string_to_list(
        curr_corp_design_subcorpus_target_language_string
    )

    # corpus_proficiency_assignment_available   : boolean
    curr_corp_proficiency_assignment_available = map_to_boolean(
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name].iloc[0][
            "corpus_proficiency_assignment_available"
        ]
    )

    # corpus_proficiency_assignment_method : list of strings
    curr_corp_proficiency_assignment_method_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_proficiency_assignment_method"]
    curr_corp_proficiency_assignment_method_list = convert_cell_string_to_list(
        curr_corp_proficiency_assignment_method_string
    )

    # corpus_learner_proficiency_assignment_instrument: list of strings
    curr_corp_learner_proficiency_assignment_instrument_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_learner_proficiency_assignment_instrument"]
    curr_corp_learner_proficiency_assignment_instrument_list = (
        convert_cell_string_to_list(
            curr_corp_learner_proficiency_assignment_instrument_string
        )
    )

    # corpus_proficiency_level: string
    curr_corp_proficiency_level_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_proficiency_level"]
    if curr_corp_proficiency_level_val == "#":
        curr_corp_proficiency_level_val = meta.NaString.NOT_AVAILABLE

    # corpus_proficiency_textAutomatic_instrument: string
    curr_corp_proficiency_textAutomatic_instrument_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_proficiency_textAutomatic_instrument"]
    if curr_corp_proficiency_textAutomatic_instrument_val == "#":
        curr_corp_proficiency_textAutomatic_instrument_val = meta.NaString.NOT_AVAILABLE

    # corpus_text_proficiency_assignment_instrument: list of strings
    curr_corp_text_proficiency_assignment_instrument_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_text_proficiency_assignment_instrument"]
    curr_corp_text_proficiency_assignment_instrument_list = convert_cell_string_to_list(
        curr_corp_text_proficiency_assignment_instrument_string
    )

    # corpus_project_contact_orig: list of strings
    curr_corp_project_contact_orig_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_project_contact_orig"]
    curr_corp_project_contact_orig_list = convert_cell_string_to_list(
        curr_corp_project_contact_orig_string
    )

    # corpus_project_duration_orig: string
    curr_corp_project_duration_orig_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_project_duration_orig"]
    if curr_corp_project_duration_orig_val == "#":
        curr_corp_project_duration_orig_val = meta.NaString.NOT_AVAILABLE
    # corpus_project_head_orig : list of strings
    curr_corp_project_head_orig_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_project_head_orig"]
    curr_corp_project_head_orig_list = convert_cell_string_to_list(
        curr_corp_project_head_orig_string
    )

    # corpus_project_institution_orig: list of strings
    curr_corp_project_institution_orig_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_project_institution_orig"]
    curr_corp_project_institution_orig_list = convert_cell_string_to_list(
        curr_corp_project_institution_orig_string
    )

    # corpus_project_type_orig: list
    curr_corp_project_type_orig_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_project_type_orig"]
    curr_corp_project_type_orig_list = convert_cell_string_to_list(
        curr_corp_project_type_orig_string
    )

    # corpus_signet_dkd: string val
    curr_corpus_signet_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_signet_dkd"]

    corpus_proj_name_orig_str = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_related_research_project_orig"]
    corpus_proj_name_orig_list = convert_cell_string_to_list(corpus_proj_name_orig_str)

    # corpus_subcorpus_target_language: string
    curr_corp_subcorpus_target_language_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_subcorpus_target_language"]
    if curr_corp_subcorpus_target_language_val == "#":
        curr_corp_subcorpus_target_language_val = meta.NaString.NOT_AVAILABLE

    # corpus_production_setting_educationalStage: list of strings
    curr_corp_production_setting_educationalStage_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_educationalStage"]
    curr_corp_production_setting_educationalStage_list = convert_cell_string_to_list(
        curr_corp_production_setting_educationalStage_string
    )

    # print(f"curr_corp_production_setting_educationalStage_list: {curr_corp_production_setting_educationalStage_list}")
    # sys.exit(-2)

    # corpus_production_setting_language_testing: string
    curr_corp_production_setting_language_testing_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_language_testing"]
    if curr_corp_production_setting_language_testing_val == "#":
        curr_corp_production_setting_language_testing_val = meta.NaString.NOT_AVAILABLE

    # corpus_production_setting_languageCourseLevel: list of strings
    curr_corp_production_setting_languageCourseLevel_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_languageCourseLevel"]
    curr_corp_production_setting_languageCourseLevel_list = convert_cell_string_to_list(
        curr_corp_production_setting_languageCourseLevel_string
    )

    # corpus_production_setting: string
    curr_corp_production_setting_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting"]
    if curr_corp_production_setting_val == "#":
        curr_corp_production_setting_val = meta.NaString.NOT_AVAILABLE

    # corpus_production_setting_naturalistic_specific: list of strings
    curr_corp_production_setting_naturalistic_specific_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_naturalistic_specific"]
    curr_corp_production_setting_naturalistic_specific_list = (
        convert_cell_string_to_list(
            curr_corp_production_setting_naturalistic_specific_string
        )
    )

    # corpus_production_setting_researchProject: boolean
    curr_corp_production_setting_researchProject_val = map_to_boolean(
        corp_meta_df[corp_meta_df["corpus_acronym"] == corpus_name].iloc[0][
            "corpus_production_setting_researchProject"
        ]
    )

    # corpus_taskLevel  : string
    curr_corp_taskLevel_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_taskLevel"]
    if curr_corp_taskLevel_val == "#":
        curr_corp_taskLevel_val = meta.NaString.NOT_AVAILABLE

    # corpus_taskType: string
    curr_corp_taskType_val = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_taskType"]
    if curr_corp_taskType_val == "#":
        curr_corp_taskType_val = meta.NaString.NOT_AVAILABLE

    # corpus_production_setting_conceptualmode: list of strings
    curr_corp_production_setting_conceptualmode_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_conceptualmode"]
    curr_corp_production_setting_conceptualmode_list = convert_cell_string_to_list(
        curr_corp_production_setting_conceptualmode_string
    )

    # corpus_mode: list of strings
    curr_corp_mode_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_mode"]
    curr_corp_mode_list = convert_cell_string_to_list(curr_corp_mode_string)

    # corpus_production_setting_participants_L1-L2-interaction: list of strings
    curr_corp_production_setting_participants_L1_L2_interaction_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_participants_L1-L2-interaction"]
    curr_corp_production_setting_participants_L1_L2_interaction_list = (
        convert_cell_string_to_list(
            curr_corp_production_setting_participants_L1_L2_interaction_string
        )
    )

    # corpus_production_setting_participants: list of strings
    curr_corp_production_setting_participants_string = corp_meta_df[
        corp_meta_df["corpus_acronym"] == corpus_name
    ].iloc[0]["corpus_production_setting_participants"]
    curr_corp_production_setting_participants_list = convert_cell_string_to_list(
        str(curr_corp_production_setting_participants_string)
    )

    # CSECTION: corpus defaults: these derive from the excel file "SWIK-LX-Metadatensheet"
    # # or from Tabelle werteübersetzung

    default_learner_proficiency_instrument = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_rating = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_ratingMethod = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_documentation = meta.NaString.NOT_AVAILABLE

    # default_learner_proficiency_estimate = meta.NaString.NOT_AVAILABLE
    # we use the below from werteübersestz
    default_learner_language_proficiency_estimateMax = "B1"
    default_learner_language_proficiency_estimateMin = "A1"

    default_learner_proficiency_Cefr = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_c_test_LevelDetail = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_c_test_percent = meta.NaString.NOT_AVAILABLE
    default_learner_proficiency_c_test_type = meta.NaString.NOT_AVAILABLE

    default_learner_majorSubject = meta.NaString.NOT_APPLICABLE
    default_learner_professionGroup = meta.NaString.NOT_APPLICABLE
    default_learner_target_language_selfAssessment = meta.NaString.NOT_AVAILABLE

    good_learner_ids = []  # ["Mo115"]

    # CSECTION document loop
    # here: docids without file extension, e.g.  SWI01_fD_Es165_c
    for ix, docid in enumerate(sorted(docid_lines)):
        if ix >= MAX_DOCS_TO_PROC:
            break
        docid = docid.strip()
        logger.info(f"Processing document with id: {docid}")

        # we create the meta data document
        learner_metadata_doc = meta.Document()

        xoutname = docid + ".v1.xml"
        xoutpath = os.path.join(outdir, xoutname)
        joutname = docid + ".v1.json"
        joutpath = os.path.join(outdir, joutname)

        comps = re.split("_", docid)
        task_id = comps[0]
        task_int = int(re.sub("SWI", "", task_id))
        school_and_text_langs = comps[1]
        school_lg = LANG_VALS_MAP.get(school_and_text_langs[0].lower())
        text_lg = LANG_VALS_MAP.get(school_and_text_langs[1].lower())
        src_learner_id = comps[2]
        p_or_c = comps[3]

        if len(good_learner_ids) > 0 and src_learner_id not in good_learner_ids:
            logger.info(
                f"Skipping learner id {src_learner_id} as it's not in the good_learner_ids list"
            )
            continue

        lrn_meta_data_match_df = learner_df[learner_df["PID"] == src_learner_id]
        if len(lrn_meta_data_match_df) == 1:
            logger.info(f"lrn_meta_data_match_df is\n{lrn_meta_data_match_df.columns}")
        elif len(lrn_meta_data_match_df) == 0:
            logger.error(
                f"we don't get any match for the learner id {src_learner_id}: {lrn_meta_data_match_df}"
            )
            continue
        else:
            logger.error(
                f"we don't get a unique match for the learner id {src_learner_id}: {lrn_meta_data_match_df}"
            )
            continue

        # CSECTION: corpus info block
        corpus_block = meta.Corpus()
        learner_metadata_doc.corpus = corpus_block

        # CSUBSECTION: admin
        ca = meta.CorpusAdministrative()
        corpus_block.administrative = ca

        ca.corpus_admin_acronym = corpus_name
        ca.corpus_admin_name = [curr_corp_name]
        ca.corpus_admin_author = []
        for caut in curr_corp_authors:
            caut = caut.replace("\xa0", " ")
            ca.corpus_admin_author.append(caut)
        ca.corpus_admin_availability = curr_corp_availability

        # Q: can we not give any info?
        ca.corpus_admin_citation_document = "notAvailable"
        ca.corpus_admin_cite_as = curr_corp_cite_as

        ca.corpus_admin_contact_mail = []
        for contemail in curr_corp_email_addresses:
            contemail = contemail.replace("\xa0", " ")
            ca.corpus_admin_contact_mail.append(contemail)

        ca.corpus_admin_contributor_dkd = DKD_CORPUS_CONTRIBS

        ca.corpus_admin_contributor_orig = []
        for cont in curr_corp_contribs_list:
            cont = cont.replace("\xa0", " ")
            ca.corpus_admin_contributor_orig.append(cont)

        ca.corpus_admin_date_of_publication = curr_corp_date_of_pub_orig

        ca.corpus_admin_documentation = curr_corp_documentation
        # TODO: check if this format is about the original corpus version or the dkd version
        ca.corpus_admin_file_format = ["CAS"]

        # Q: really no info for SWIKO/WTLD? tabelle werteübersetzung has "#"
        # Q: also, is thisthe licence of the source corpus? if yes, then it seems we're not saying anything about 
        # our own licence?
        ca.corpus_admin_licence = curr_corp_licence
        ca.corpus_admin_licence_fulltext = curr_corp_licenceFulltext
        ca.corpus_admin_licence_url = curr_corp_licence_url

        ca.corpus_admin_other_versions = []
        for item in curr_corp_other_versions_list:
            item = item.replace("\xa0", " ")
            if item == "#":
                item = meta.NaString.NOT_AVAILABLE
            ca.corpus_admin_other_versions.append(item)

        ca.corpus_admin_pid_dkd = curr_corp_pid_dkd
        ca.corpus_admin_pid_orig = curr_corp_pid_orig

        ca.corpus_admin_ref_article = curr_corp_ref_article

        # TODO: i picked a ref that's mentioned in the SWIKO manual. In the corpora table, there's actually no entry for SWIKO.
        # Should we really have no reference here?
        ca.corpus_admin_references_other = [REF1]
        # comment in the below if we want to strictly take this info  from the excel table for corpora
        # ca.corpus_admin_references_other = curr_corps_referencesOther_list

        ca.corpus_admin_research_paper = curr_corp_researchPapers_list
        ca.corpus_admin_url_download = curr_corp_URL_download_list

        # FIXME: check this next: should we use this?
        ca.corpus_admin_urlquery = "https://ifm-swiko.unifr.ch/?view=login"
        ca.corpus_admin_version_orig = "notAvailable"

        # CSUBSECTION: design
        cd = meta.CorpusDesign()
        corpus_block.design = cd

        cd.corpus_design_description = "notAvailable"
        # Q:  in der Tabelle für SWIKO war eine Spalte longitudinal , die nur ja/nein als Werte hat.
        # Aber theoretisch hatten wir mal "pseudo-longitudinal" als dritte Option.
        # Ist diese Option nun eliminiert worden bzw. ist pseudo-longitudinal mit longitudinal gemergt worden?
        if curr_corp_designType == True:
            cd.corpus_design_design_type = meta.StudyDesign.LONGITUDINAL
        else:
            cd.corpus_design_design_type = meta.StudyDesign.CROSS_SECTIONAL

        cd.corpus_design_group = curr_corp_corpusGroup
        cd.corpus_design_is_comparable_data_included = (
            curr_corp_comparable_data_included
        )

        cd.corpus_design_l1_language = curr_corp_design_l1_language_val
        cd.corpus_design_l1_type = curr_corp_design_l1_type_val

        # TODO: this only gets a default value right now
        cs = meta.CorpusDesign.CorpusDesignSize(value=-1.0, unit="KB")
        cd.corpus_design_size = cs

        cd.corpus_design_target_language = curr_corp_design_target_language_list
        cd.corpus_design_target_language_type = (
            curr_corp_design_target_language_type_list
        )


        curr_corp_design_time_coll = corp_meta_df[
            corp_meta_df["corpus_acronym"] == corpus_name
        ].iloc[0]["corpus_time_of_data_collection"]
        cd.corpus_design_time_of_data_collection = curr_corp_design_time_coll


        # CSUBSECTION: proficiency
        cp = meta.CorpusProficiency()
        corpus_block.proficiency = cp

        cp.corpus_proficiency_assignment_method = (
            curr_corp_proficiency_assignment_method_list
        )
        cp.corpus_proficiency_is_assignment_available = (
            curr_corp_proficiency_assignment_available
        )
        cp.corpus_proficiency_learner_assignment_instrument = (
            curr_corp_learner_proficiency_assignment_instrument_string
        )
        cp.corpus_proficiency_level = curr_corp_proficiency_level_val
        # TODO: check this
        cp.corpus_proficiency_text_assignment_instrument = (
            curr_corp_text_proficiency_assignment_instrument_list
        )

        cp.corpus_proficiency_text_automatic_assignment_instrument = (
            curr_corp_proficiency_textAutomatic_instrument_val
        )

        # CSUBSECTION: project
        cj = meta.CorpusProject()
        corpus_block.project = cj

        cj.corpus_project_contact_orig = curr_corp_project_contact_orig_list
        cj.corpus_project_duration_dkd = "2022-2025"

        cj.corpus_project_duration_orig = curr_corp_project_duration_orig_val
        cj.corpus_project_head_dkd = ["Torsten Zesch", "Katrin Wisniewski"]

        cj.corpus_project_head_orig = curr_corp_project_head_orig_list
        cj.corpus_project_institution_dkd = [
            "FernUniversität in Hagen",
            "Universität Leipzig",
        ]

        cj.corpus_project_institution_orig = curr_corp_project_institution_orig_list
        cj.corpus_project_name_dkd = (
            meta.DkdProjectName.DATENKOMPETENZEN_IN_DA_F_DA_Z_EXPLORATION_SPRACHTECHNOLOGISCHER_ANS_TZE_ZUR_ANALYSE_VON_L2_ERWERBSSTUFEN_IN_LERNERKORPORA_DES_DEUTSCHEN
        )

        cj.corpus_project_name_orig = corpus_proj_name_orig_list
        cj.corpus_project_type_dkd = (
            "Bundesministerium für Bildung und Forschung (BMBF)"
        )
        cj.corpus_project_type_orig = curr_corp_project_type_orig_list

        # TODO: this can only be done later unless we make up rules about the repository domain in advance
        cj.corpus_project_url_dkd = meta.NaString.NOT_AVAILABLE

        # CSUBSECTION: project
        csub = meta.CorpusSubcorpus()
        corpus_block.subcorpus = csub

        # TODO: fill quantitative fields later
        # NB: we use 1234567890 as our default placeholder value for ints for now
        csub.corpus_subcorpus_signet = curr_corpus_signet_val
        csub.corpus_subcorpus_size_learners = 1234567890
        csub.corpus_subcorpus_size_texts = 1234567890
        csub.corpus_subcorpus_size_tokens = 1234567890

        csub.corpus_subcorpus_target_language = curr_corp_subcorpus_target_language_val

        # CSECTION: production setting

        prod_set = meta.ProductionSetting()
        learner_metadata_doc.production_setting = prod_set

        prod_set.production_setting_school_grade = lrn_meta_data_match_df[
            "schoolyear"
        ].iloc[0]

        prod_set.production_setting_educational_stage = (
            curr_corp_production_setting_educationalStage_list
        )
        prod_set.production_setting_language_test = (
            curr_corp_production_setting_language_testing_val
        )

        prod_set.production_setting_language_course_level = (
            curr_corp_production_setting_languageCourseLevel_list
        )
        prod_set.production_setting_naturalistic = (
            curr_corp_production_setting_naturalistic_specific_list
        )
        prod_set.production_setting_collected_in_research_project = (
            curr_corp_production_setting_researchProject_val
        )
        prod_set.production_setting_setting = curr_corp_production_setting_val

        # CSECTION: task
        task_block = meta.TaskBlock()
        learner_metadata_doc.task = task_block

        task_block.task_id = meta.NaString.NOT_AVAILABLE
        task_block.task_id_orig = task_id
        task_block.task_title = meta.NaString.NOT_AVAILABLE
        task_block.task_comparison = meta.NaString.NOT_AVAILABLE

        # TODO: is there info on task descriptions? the mindmap seems to point to the cmslc var task_instructions
        task_block.task_description = meta.NaString.NOT_AVAILABLE
        task_block.task_description_detailed = meta.NaString.NOT_AVAILABLE

        # TODO: we have no info on duration, right?
        task_block.task_duration_minutes = meta.NaString.NOT_AVAILABLE

        # TODO: check this next: is the right kind of info to put in this field?
        task_block.task_instructions = look_up_swiko_task_description(task_id, task_df)

        # TODO: we have no info on duration, do we?
        task_block.task_is_duration_limited = meta.NaString.NOT_AVAILABLE
        task_block.task_level = curr_corp_taskLevel_val


        # TODO: check this next: is this specifiable for wtld? should be yes or no.
        # couldn't find a good column in the Corpora table.
        task_block.task_official_language_test = meta.NaString.NOT_AVAILABLE
        task_block.task_official_language_test_specific = meta.NaString.NOT_AVAILABLE

        # TODO: i think the corpora table has no info on these three:
        task_block.task_options = meta.NaString.NOT_AVAILABLE
        task_block.task_stimulus_offered = meta.NaString.NOT_AVAILABLE
        task_block.task_stimulus_type = meta.NaString.NOT_APPLICABLE

        # CSUBSECTION: task interaction
        ti_block = meta.InteractionBlock()
        task_block.interaction = ti_block

        # TODO: check: is this the right kind of info to put in this field?
        ti_block.task_interaction_conceptual_mode = (
            curr_corp_production_setting_conceptualmode_list
        )

        ti_block.task_interaction_expected_rhetorical_functions = [
            meta.NaString.NOT_AVAILABLE
        ]
        ti_block.task_interaction_formality = meta.NaString.NOT_AVAILABLE

        tim = meta.TaskInteractionMode()
        ti_block.task_interaction_mode = tim
        tim.corpus_mode = curr_corp_mode_list
        # TODO: check this next: is this the right kind of info to put in this field? 
        # It's not in the corpora table!
        tim.situation_mode = meta.DataProductionSettingMode.WRITTEN

        ti_block.task_interaction_participants_l1_l2_interaction = (
            curr_corp_production_setting_participants_L1_L2_interaction_list
        )
        # TODO: check if this is the right kind of info to put in this field?
        ti_block.task_interaction_participants = (
            curr_corp_production_setting_participants_list
        )

        # TODO: hat monologisches L2 einen "Interaktionstyp?"
        # aber interacton _type scheint CMSLC situation_register  und damit den SWIKO-Tasks zu entsprechen?
        ti_block.task_interaction_type = meta.NaString.NOT_AVAILABLE

        # CSECTION: LEARNER info

        learner_block = meta.Learner()
        learner_metadata_doc.learner = learner_block
        # TODO: what should the dkd learner id look like?
        learner_block.learner_id = meta.NaString.NOT_AVAILABLE
        learner_block.learner_id_orig = src_learner_id

        # how should I count this? do we assume that the info on languages is exhaustive per learner?
        learner_block.learner_l_count = meta.NaString.NOT_AVAILABLE

        # TODO: is it  ok to look at columns l1a - l1c for this
        if count_l1s(lrn_meta_data_match_df.iloc[0], ["l1a", "l1b", "l1c"]) > 1:
            learner_block.learner_multiple_l1 = True
        else:
            learner_block.learner_multiple_l1 = False

        learner_block.learner_text_count = len(learner_2_taskIds_map[src_learner_id])
        learner_block.learner_note = meta.NaString.NOT_AVAILABLE

        # CSUBSECTION: learner sociodemographics
        learner_socio = meta.Sociodemographics()
        learner_block.sociodemographic = learner_socio

        # we're not making any guesses/inferences about the country of birth
        learner_socio.learner_socio_birthplace = meta.NaString.NOT_AVAILABLE

        learner_socio.learner_socio_country = "CHE"

        # TODO: don't we know anything about this next?
        # it's not in the corpora table and the wereübersetzungstabelle has no info on it either
        learner_socio.learner_socio_educational_background = meta.NaString.NOT_AVAILABLE

        gdrval = str(lrn_meta_data_match_df.iloc[0]["gender"])
        if gdrval not in GENDER_MAP:
            mapped_gdr_val = meta.NaString.NOT_AVAILABLE
        else:
            mapped_gdr_val = GENDER_MAP[gdrval]

        learner_socio.learner_socio_gender = mapped_gdr_val

        learner_socio.learner_socio_major_subject = default_learner_majorSubject

        learner_socio.learner_socio_profession = default_learner_professionGroup

        learner_socio.learner_socio_school_grade = lrn_meta_data_match_df.iloc[0][
            "schoolyear"
        ]

        # CSUBSECTION: learner language

        # Q: what is the relationship to text_lg and school_lg in the **file name**? 
        # Is it completely independent?
        # I think text language == target language
        # if text lang is deu and deu is not in ["l1a", "l1b", "l1c"], then we have deu as L2 and targt language, innit?

        learner_block.language = []
        canton = lrn_meta_data_match_df.iloc[0]["canton"].lower()

        # CSUBSUBSECTION: L1(s)
        lgs_map = {"deu": "Target language, L2"}
        l1s = []
        for lan in ["l1a", "l1b", "l1c"]:
            lan_string = lrn_meta_data_match_df.iloc[0][lan]
            if (
                lan_string not in ["nan", "NA"]
                and lan_string != None
                and pd.isna(lan_string) == False
            ):
                lgs_map[lan_string] = "L1"
                l1s.append(lan_string)

        for lan_string in lgs_map.keys():
            language_status_list = re.split("\s*,\s*", lgs_map[lan_string])
            logger.info(f"Creating language entry for {lan_string}")

            # NB: create the Language entry for the current language
            learner_language = meta.LanguageOfSpeaker()
            learner_block.language.append(learner_language)

            lg_name_en, lg_name_de = get_lan_info_for_iso_code(lan_string, lang_info_df)

            my_lang = meta.Language(
                name_en=lg_name_en, value=lg_name_de, iso_code_639_3=lan_string
            )
            learner_language.learner_language_iso639_3 = my_lang

            # NB: lang status  can be L1, L2, or Target language
            learner_language.learner_language_status = language_status_list
            learner_language.learner_language_dominant_word_order = lookup_word_order(
                lang_info_df, lan_string
            )

            # NB: we're following info in werteübsetzung
            curr_los = LANG_VALS_MAP.get(
                lrn_meta_data_match_df.iloc[0]["language_of_schooling"]
            )
            if curr_los in lan_string:
                learner_language.learner_language_is_spoken_school = True
            else:
                learner_language.learner_language_is_spoken_school = False

            # Q: are we making any assumptions about the home lanuage?
            # werteübsetzung provides no default; student metadata has no info , AFAIK
            learner_language.learner_language_is_spoken_home = (
                meta.NaString.NOT_AVAILABLE
            )
            learner_language.learner_language_parent_l1 = meta.NaString.NOT_AVAILABLE

            if "Target language" in language_status_list:
                # CSUBSUBSECTION: learner language exposure
                lle = meta.LanguageExposure()
                learner_language.exposure = lle

                try:
                    curr_age = float(
                        lrn_meta_data_match_df.iloc[0]["text_learner_ageProduction"]
                    )
                except:
                    curr_age = meta.NaString.NOT_AVAILABLE
                logger.info(f"Age: {curr_age}")

                raw_doi = lrn_meta_data_match_df.iloc[0][
                    "learner_language_exposure_durationOfInstruction"
                ]
                try:
                    curr_doi = float(raw_doi)
                except:
                    curr_doi = meta.NaString.NOT_AVAILABLE
                logger.info(f"Duration of Instruction: {curr_doi} {raw_doi}")
                logger.info(f"type check {str(type(curr_age))} {str(type(curr_doi))}")

                # TODO the learner metadata has learner_language_exposure_ageOfOnset:  
	            # should we use that or do the calculation below?
                logger.info(f"school language: {school_lg}")
                if (
                    "str" not in str(type(curr_age))
                    and "NaString" not in str(type(curr_age))
                    and "NaString" not in str(type(curr_doi))
                    and "str" not in str(type(curr_doi))
                ):
                    if school_lg == "fra":
                        lle.learner_language_exposure_onset = (
                            (curr_age * 12) - curr_doi
                        ) / 12
                    elif school_lg == "deu":
                        lle.learner_language_exposure_onset = (
                            meta.NaString.NOT_APPLICABLE
                        )
                    else:
                        lle.learner_language_exposure_onset = (
                            meta.NaString.NOT_AVAILABLE
                        )
                else:
                    logger.error(
                        f"either age {curr_age} or learner_language_exposure_durationOfInstruction {curr_age} is not a number, skipping this entry"
                    )
                    lle.learner_language_exposure_onset = meta.NaString.NOT_AVAILABLE
                    curr_age = meta.NaString.NOT_AVAILABLE
                    curr_doi = meta.NaString.NOT_AVAILABLE

                # TODO: Wertebereich festlegen für onset_group; basiered auf learner_target_language_onset aus student metadata
                lle.learner_language_exposure_onset_group = meta.NaString.NOT_AVAILABLE

                # alle Ids mit "RI..." => 6, alle IDs mit "Mo..." und "ES..." => 5, alle anderen "notApplicable"
                if src_learner_id.lower().startswith(
                    "ri"
                ):  # and canton.lower() == "ri":
                    lle.learner_language_exposure_duration_of_instruction = 6
                elif src_learner_id.lower().startswith(
                    "mo"
                ):  # and canton.lower() == "mo":
                    lle.learner_language_exposure_duration_of_instruction = 5
                elif src_learner_id.lower().startswith(
                    "es"
                ):  # and canton.lower() == "es":
                    lle.learner_language_exposure_duration_of_instruction = 5
                else:
                    lle.learner_language_exposure_duration_of_instruction = (
                        meta.NaString.NOT_APPLICABLE
                    )

                lle.learner_language_exposure_duration_of_use = (
                    meta.NaString.NOT_AVAILABLE
                )

                # IF los=f => "mainly in controlled teaching contexts" IF ELSE "mainly without controlled teaching processes"
                if school_lg == "f":
                    lle.learner_language_exposure_input = (
                        "mainly in controlled teaching contexts"
                    )
                else:
                    lle.learner_language_exposure_input = (
                        "mainly without controlled teaching processes"
                    )

                # IF los="f" -> "school" ; IF l1a!="deu" und los="deu" -> "notApplicable"; IF l1a="deu" und los="deu" -> notApplicable
                if school_lg == "fra":
                    lle.learner_language_exposure_institution = ["school"]
                elif school_lg == "deu" and "deu" not in l1s:
                    lle.learner_language_exposure_institution = [
                        meta.NaString.NOT_APPLICABLE
                    ]
                elif "deu" in l1s and "deu" in l1s:
                    lle.learner_language_exposure_institution = [
                        meta.NaString.NOT_APPLICABLE
                    ]
                else:
                    lle.learner_language_exposure_institution = [
                        meta.NaString.NOT_AVAILABLE
                    ]

                lle.learner_language_exposure_months_spent_environment = (
                    meta.NaString.NOT_AVAILABLE
                )

                # {EINFÜGEN:[learner_target_language_onset]-EINFÜGEN:[text_learner_ageProduction]: instructed}
                lle.learner_language_exposure_learning_context = (
                    f"{lle.learner_language_exposure_onset}-{curr_age}: instructed"
                )

                lle.learner_language_exposure_place_acquisition = "CHE"

                stay_in_env = map_to_boolean(
                    lrn_meta_data_match_df.iloc[0][
                        "learner_language_exposure_WasInEnvironment"
                    ]
                )
                lle.learner_language_exposure_was_in_environment = stay_in_env

                # the corpus table has no info, the learner table has auxiliary variable learner language exposure input
                llexp_input = lrn_meta_data_match_df.iloc[0][
                    "learner_language_exposure_input"
                ]
                if "mainly without" in llexp_input:
                    lle.learner_language_was_instructed = False
                else:
                    lle.learner_language_was_instructed = True

                # CSUBSUBSECTION: learner language proficiency
                llp = meta.LearnerLanguageProficiency()
                learner_language.proficiency = llp

                # Q: the excelfile also has a var `default_learner_proficiency_rating` but I'm not sure if that should be mapped to any of the schema vars, especially the score variable below
                llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE

                llp.learner_language_proficiency_cefr = default_learner_proficiency_Cefr

                #  the learner excel table has no mapping for this next set of variables
                # we use defaults from the werteübersetzungstabelle
                llp.learner_language_proficiency_c_test_cefr = (
                    meta.NaString.NOT_AVAILABLE
                )

                llp.learner_language_proficiency_c_test_percent = (
                    default_learner_proficiency_c_test_percent
                )
                llp.learner_language_proficiency_c_test_level_detail = (
                    default_learner_proficiency_c_test_LevelDetail
                )
                llp.learner_language_proficiency_c_test_type = (
                    default_learner_proficiency_c_test_type
                )

                # we  get  default estimates from werteübersetzungstabelle
                llp.learner_language_proficiency_estimate_min = (
                    default_learner_language_proficiency_estimateMin
                )
                llp.learner_language_proficiency_estimate_max = (
                    default_learner_language_proficiency_estimateMax
                )

                llp.learner_language_proficiency_self_assessment = (
                    default_learner_target_language_selfAssessment
                )

                llp.learner_language_proficiency_assignment_instrument = (
                    default_learner_proficiency_instrument
                )
                llp.learner_language_proficiency_assignment_method = (
                    default_learner_proficiency_ratingMethod
                )

                llp.learner_language_proficiency_documentation = (
                    default_learner_proficiency_ratingMethod
                )

        # CSECTION: TEXT related info
        text_block = meta.TextProperties()
        learner_metadata_doc.text = text_block

        # Q: is xml or csv more "canonical"? or should this be our verison
        text_block.text_file = docid + ".xml"

        # TODO: what is the dakoda id?
        text_block.text_id = meta.NaString.NOT_AVAILABLE

        # Q: is there a specific text id, like the filename minus the extension?
        text_block.text_id_orig = meta.NaString.NOT_AVAILABLE

        # TODO: we need to create an instance of meta.Language here  from the three letter lang code
        txt_lg_name_en, txt_lg_name_de = get_lan_info_for_iso_code(
            text_lg, lang_info_df
        )

        text_lang = meta.Language(
            name_en=txt_lg_name_en, value=txt_lg_name_de, iso_code_639_3=lan_string
        )
        text_block.text_language = text_lang

        # Q: we get the int part of the task id ? or do we only consider the tasks that the speaker has actually completed?
        # the task int is taken out of the filename
        text_block.text_longitudinal_order = task_int

        # TODO: check if we do have this info after all:  Werteübersetzung says the students metadata should have info
        text_block.text_time_of_creation = lrn_meta_data_match_df["yearcollected"].iloc[
            0
        ]

        text_block.text_token_count = 1234567890
        text_block.text_clause_count = 1234567890

        # Q: in the swiko manual on page 9, the tasks are assigned to topics: should we use this? mabe as a note?
        text_block.text_topic_autom = meta.NaString.NOT_AVAILABLE

        text_block.text_note = meta.NaString.NOT_AVAILABLE

        # CSUBSECTION: text learner
        text_learner = meta.TextLearner()
        text_block.learner = text_learner

        # NB: not using the `age` column but the col with rounded values.
        text_learner.text_learner_age_production = lrn_meta_data_match_df[
            "text_learner_ageProduction"
        ].iloc[0]
        # CHECK: what does aggregated mean? does DKD have predefined age-ranges?
        text_learner.text_learner_age_production_aggregated = (
            meta.NaString.NOT_AVAILABLE
        )
        text_learner.text_learner_role = meta.NaString.NOT_AVAILABLE

        # CSUBSECTION: text proficiency
        tprof = meta.TextProficiency()
        text_block.proficiency = tprof

        tprof.text_proficiency_assignment_instrument = meta.NaString.NOT_AVAILABLE
        tprof.text_proficiency_cefr = meta.NaString.NOT_AVAILABLE
        tprof.text_proficiency_cefr_autom = meta.NaString.NOT_AVAILABLE
        tprof.text_proficiency_documentation = meta.NaString.NOT_AVAILABLE
        tprof.text_proficiency_official_language_testing_score = (
            meta.NaString.NOT_AVAILABLE
        )
        tprof.text_proficiency_score = meta.NaString.NOT_AVAILABLE

        # CSUBSECTION: text annotation
        txtanno = meta.TextAnnotation()
        text_block.annotation = txtanno

        # I believe that no, Swiko doesn't have borrowings
        txtanno.text_annotation_borrowed = meta.NaString.NOT_AVAILABLE

        # Q: The original data has some error anno and ths  (cf. SWIKO_Wertebnersetzung
        # but we're not carrying that over.
        txtanno.text_annotation_has_error_annotation = meta.NaString.NOT_AVAILABLE
        txtanno.text_annotation_has_target_hypotheses = meta.NaString.NOT_AVAILABLE

        # CSECTION: annotation info block
        # NB: this anno block is about the dkd version!
        # Q: should this whole block repeat? does every annotation layer get such a block? i.e. pos, lemma etc?
        anno_block = meta.Annotation()
        learner_metadata_doc.annotation = anno_block
        anno_block.annotation_automatic = True
        anno_block.annotation_corrected = False
        # TODO: should this next be a url?
        anno_block.annotation_documentation = meta.NaString.NOT_AVAILABLE

        anno_block.annotation_evaluation = meta.NaString.NOT_AVAILABLE
        # since we have multiple annotation tools, should we wrap this?
        anno_block.annotation_tool = meta.NaString.NOT_AVAILABLE
        anno_block.annotation_tool_version = meta.NaString.NOT_AVAILABLE
        anno_block.annotation_type = meta.NaString.NOT_AVAILABLE

        # CSECTION: annotator info block
        # Q:  do we use this for souroce project annotators?
        # in theory, we could specify annotators per layer. i.e. the transcriber pertains to our ctok (base) layer but not to the TH layers.
        # since our layers are automatic, we don't not have any (manual) annotator.
        annot = meta.Annotator()
        learner_metadata_doc.annotator = annot

        # TODO: using trancriber sigle from transcriber table here.
        # If "checked" is different from "transcriber", do i treat both as annotators?
        #  right now i'm only using the transcriber info
        file_id = task_id + "_" + school_and_text_langs + "_" + src_learner_id
        logger.info("looking up anotator for file_id: " + file_id)
        if file_id in doc2trans_map:
            annot.annotator_id = doc2trans_map[file_id]
        else:
            annot.annotator_id = meta.NaString.NOT_AVAILABLE

        annot.annotator_l1 = meta.NaString.NOT_AVAILABLE
        annot.annotator_l2 = meta.NaString.NOT_AVAILABLE
        annot.annotator_note = meta.NaString.NOT_AVAILABLE
        annot.annotator_target_language_competence = meta.NaString.NOT_AVAILABLE
        annot.annotator_type = meta.NaString.NOT_AVAILABLE

        # CSECTION: write document
        with open(xoutpath, "w", encoding="utf-8") as audi:
            audi.write(serializer.render(learner_metadata_doc))
        with open(joutpath, "w", encoding="utf-8") as jaudi:
            jaudi.write(jserializer.render(learner_metadata_doc))


if __name__ == "__main__":
    main()
