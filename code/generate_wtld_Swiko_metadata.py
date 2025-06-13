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

# REVIEW: update to most recent schema here!
from dkd_meta import dakoda_metadata_scheme as meta

# import xsdata.formats.dataclass.serializer.XmlSerializer as XmlSerializer
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers import JsonSerializer


jserializer = JsonSerializer()

from xsdata.formats.dataclass.serializers.config import SerializerConfig


# REVIEW: path to xsd
serializer = XmlSerializer(
	config=SerializerConfig(
		xml_declaration=False,
		pretty_print=True,
		ignore_default_attributes=False,
		schema_location="/mnt/biggie/dakoda_metadaten/xml_json_schema_2025/dakoda_metadata_scheme_2025-06-10.xsd",
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

	lang_df = lang_info_df[lang_info_df["learner_language_iso6393"] == lan_string]
	if len(lang_df) == 1:
		lang_row = lang_df.iloc[0]
		lan_name_en = lang_row["learner_language_iso6393_en"]
		lan_name_de = lang_row["learner_language_iso6393_de"]
		lan_group = lang_row["learner_language_group"]
	else:
		logger.error(f"could not retrieve info for language with code {lan_string}")
		lan_name_en = meta.NaString.NOT_AVAILABLE
		lan_name_de = meta.NaString.NOT_AVAILABLE
		lan_group = meta.NaString.NOT_AVAILABLE
	return (lan_name_en, lan_name_de, lan_group)


#  in_DAKODA    L1_DKD  L2_DKD  Orig_Wert_deutscheBezeichnung   Orig_Wert_englischeBezeichnung
#  Orig_code_ISO    Orig_code_anderer   iso639-3_de iso639-3_en     iso639-3_code   dominantWordOrder


def lookup_word_order(df: pd.DataFrame, code: str) -> str:
	logger.info(f"Looking up word order for ISO code {code}")
	match_df = df[df["learner_language_iso6393"] == code]
	if len(match_df) == 0:
		return "notAvailable"
	else:
		assert len(match_df) == 1
		return match_df.iloc[0]["learner_language_dominantWordOrder"]


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


"""


	Dakoda_Schema Operatoren
	* berechnen: Hier berechnest du automatisch die Anzahl der Texte, die Größe der Datei oder das automatische CEFR-Level
	* eintragen: es gibt für alle Texte nur den einen Wert in der Spalte  "value definition"
	* ermitteln: in Spalte J "ermitteln" gibt es einen Auftrag für dich (ab Version 2025-06-11 im excel!)
	* extra: ich versuch dir nächste Woche für alle diese Variablen Werte für alle Korpora in Extra-Tabellen zu schicken. 
		* Bei ANNOTATION oder ANNOTATOR setze ich dir den Hut auf. Sag Bescheid, wenn wir dir da etwas formulieren sollen!
	* matchen: Falls es Infos in den Metadaten gibt, schreiben wir sie in die learner/text-Werteübersetzung, 
		falls nicht, tauchen sie in der Werteübersetzung nicht auf. 
		Dann schreibst du den den "Default-Wert" bei Fehlwerten hinein
	* recherchieren: sollte zwingend immer in der Werteübersetzung stehen

"""


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

	doc_stats_df = pd.read_excel(
		config["corpus"]["doc_stats_table"], engine="openpyxl", header=0
	)
	print(doc_stats_df.head())
	
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
		logger.info(f"collecting info from docid {docid}")
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
		curr_corp_proficiency_level_valMin = "A1"
		curr_corp_proficiency_level_valMax = "B1"


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

	# FIXME
	if curr_corp_taskLevel_val == "#":
		curr_corp_taskLevel_valMax = meta.NaString.NOT_AVAILABLE
		curr_corp_taskLevel_valMin = meta.NaString.NOT_AVAILABLE


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

	default_learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_instrument = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_rating = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_ratingMethod = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_documentation = meta.NaString.NOT_AVAILABLE

	# default_learner_proficiency_estimate = meta.NaString.NOT_AVAILABLE
	# we use the below from werteübersestz
	default_learner_language_proficiency_estimateMax = "B1"
	default_learner_language_proficiency_estimateMin = "A1"

	default_learner_proficiency_Cefr = meta.NaString.NOT_AVAILABLE
	default_learner_language_proficiency_cTestCefr = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_c_test_LevelDetail = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_c_test_percent = meta.NaString.NOT_AVAILABLE
	default_learner_proficiency_c_test_type = meta.NaString.NOT_AVAILABLE

	# major subject is at university level, not at the school level
	# profession ~ job is also for adults only
	# inference rules for both:
	# notAvailable; if productionSetting_educationalStage=[early childhood, primary, lower secondary,
	# upper secondary, post-secondary non-tertiary, short-cycle tertiary] -> notApplicable
	default_learner_majorSubject = meta.NaString.NOT_APPLICABLE
	default_learner_professionGroup = meta.NaString.NOT_APPLICABLE

	default_learner_target_language_selfAssessment = meta.NaString.NOT_AVAILABLE

	good_learner_ids = [] #  # "Pf260"]  # ["Mo115"]

	# CSECTION document loop
	# here: docids without file extension, e.g.  SWI01_fD_Es165_c
	for ix, docid in enumerate(sorted(docid_lines)):
		if ix >= MAX_DOCS_TO_PROC:
			break
		docid = docid.strip()
		logger.info(f"Processing {ix}-th document with id: {docid}")

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
		curr_students_school_lg = LANG_VALS_MAP.get(school_and_text_langs[0].lower())
		text_lg = LANG_VALS_MAP.get(school_and_text_langs[1].lower())
		src_learner_id = comps[2]
		p_or_c = comps[3]

		if len(good_learner_ids) > 0 and src_learner_id not in good_learner_ids:
			logger.info(
				f"Skipping learner id {src_learner_id} as it's not in the good_learner_ids list"
			)
			continue

		lrn_meta_data_match_df = learner_df[learner_df["PID"] == src_learner_id]
		for col in lrn_meta_data_match_df.columns:
			logger.info(f"field: {col} {lrn_meta_data_match_df[col]}\n")
		logger.info(f"lrn_meta_data_match_df\n{lrn_meta_data_match_df}")
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
		# Q: also, is thisthe licence of the source corpus?
		# if yes, then it seems we're not saying anything about our own licence?
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
		# FIXME take from table
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
		# CURSOR: 
		cp.corpus_proficiency_level_max = curr_corp_proficiency_level_valMax
		cp.corpus_proficiency_level_min = curr_corp_proficiency_level_valMin
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

		# for swiko, we can look this  up.
		#
		# If we had missing data, we would do inference as follows:
		# IF productionSetting_educationalStage=[early childhood, primary, lower secondary, upper secondary,
		# post-secondary non-tertiary, short-cycle tertiary] -> notAvailable;  ELSE -> notApplicable
		#
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

		# TODO: check this next: is it the right kind of info to put in this field?
		task_block.task_instructions = look_up_swiko_task_description(task_id, task_df)

		# TODO: we have no info on duration, do we?
		task_block.task_is_duration_limited = meta.NaString.NOT_AVAILABLE
		task_block.task_level_max = curr_corp_taskLevel_valMax
		task_block.task_level_min = curr_corp_taskLevel_valMin

		# TODO: check this next: is this specifiable for wtld? should be yes or no.
		# couldn't find a good column in the Corpora table.
		task_block.task_official_language_test = meta.NaString.NOT_AVAILABLE
		task_block.task_official_language_test_specific = meta.NaString.NOT_AVAILABLE

		# TODO: i think the corpora table has no info on these three:
		task_block.task_options = meta.NaString.NOT_AVAILABLE
		task_block.task_stimulus_offered = meta.NaString.NOT_AVAILABLE
		task_block.task_stimulus_type = meta.NaString.NOT_APPLICABLE

		# TODO Fix this ; we no longer have a block like that!! Mi 11 Jun 2025 ( KW23 )
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

		# TODO: so far no task table
		ti_block.task_interaction_mode = meta.NaString.NOT_AVAILABLE       
		
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

		# Q: what is the relationship to text_lg and curr_students_school_lg in the **file name**?
		# Is it completely independent?
		# I think text language == target language
		# if text lang is deu and deu is not in ["l1a", "l1b", "l1c"], then we have deu as L2 and targt language, innit?

		learner_block.language = []
		canton = lrn_meta_data_match_df.iloc[0]["canton"]
		if pd.isna(canton) == False:
			canton = canton.lower()
		else:
			canton = meta.NaString.NOT_AVAILABLE

		learners_spoken_school_language_str = lrn_meta_data_match_df.iloc[0][
			"learner_language_isSpokenSchool"
		]
		learners_spoken_school_language_list = convert_cell_string_to_list(
			learners_spoken_school_language_str
		)

		# CSUBSUBSECTION: collect langs
		# FIXME: check if language of schooling , if not DEU, is among known L1 languages
		# if not add such languages to the lang list but with an empty status "notAvailable"
		l1s = []
		lgs_map = {}
		for lan in ["l1a", "l1b", "l1c"]:
			lan_string = lrn_meta_data_match_df.iloc[0][lan]
			if (
				lan_string not in ["nan", "NA"]
				and lan_string != None
				and pd.isna(lan_string) == False
			):
				lgs_map[lan_string] = "L1"
				l1s.append(lan_string)

		# if German is among the L1s also mark it as a Target language
		if "deu" in lgs_map:
			lgs_map["deu"] = "Target language, L1"
		# if not add it as an L2 and Target language
		else:
			lgs_map["deu"] = "Target language, L2"

		logger.info(f"languages: {l1s} {lgs_map}")

		logger.info(
			f"School language: {curr_students_school_lg} {lgs_map.get(curr_students_school_lg)}"
		)

		# NB: we make sure that the school language is listed among the student's language
		# even if it is not an L1 and  not German (we already added German above)
		if curr_students_school_lg not in lgs_map:
			lgs_map[curr_students_school_lg] = "notAvailable"

		PROJ_LANGS = ["deu", "fra", "eng", "ger"]

		# CSUBSUBSECTION: iterate across all languages (i.e. L1s and L2s )
		for lan_string in lgs_map.keys():
			curr_language_status_list = re.split("\s*,\s*", lgs_map[lan_string])
			logger.info(f"Creating language entry for {lan_string}")

			# NB: create the Language entry for the current language
			learner_language = meta.LanguageOfSpeaker()
			learner_block.language.append(learner_language)

			lg_name_en, lg_name_de, lg_group = get_lan_info_for_iso_code(
				lan_string, lang_info_df
			)

			my_lang = meta.Language(
				name_en=lg_name_en,
				value=lg_name_de,
				iso_code_639_3=lan_string,
				group=lg_group,
			)
			learner_language.learner_language_iso639_3 = my_lang

			# NB: set the language status:  can be one L1, L2, or Target language. In case of missing data "notAvailable"
			learner_language.learner_language_status = curr_language_status_list


			# REVISIT
			if "Target language"  in curr_language_status_list:
				learner_language.learner_language_is_target = True 
			else:
				learner_language.learner_language_is_target = False

			learner_language.learner_language_dominant_word_order = lookup_word_order(
				lang_info_df, lan_string
			)


			learner_language.learner_language_group = lg_group
			# NB: we're following info in werteübsetzung
			curr_language_of_schooling = LANG_VALS_MAP.get(
				lrn_meta_data_match_df.iloc[0]["language_of_schooling"]
			)

			# for cases of missing data, the default value to be used is "notAvailable"
			#
			# here we can check if the current language is the languageg of schooling;
			# if so we treat it as spoken
			# NB: this should match the entry in the students' metadata table:
			# if current lg is identical to  "learner_school_language" for teh student, then that lg is spoken at school
			logger.info(
				f"curr language of schooling: {curr_language_of_schooling} vs. curr lg {lan_string} vs learners_spoken_school_language_list {learners_spoken_school_language_list}"
			)
			if lan_string in curr_language_of_schooling:
				learner_language.learner_language_is_spoken_school = True
				assert lan_string in learners_spoken_school_language_list
			else:
				learner_language.learner_language_is_spoken_school = False
				assert lan_string not in learners_spoken_school_language_list

			# TODO: are we making any assumptions about the home lanuage?
			# werteübsetzung provides no default; student metadata has no info , AFAIK
			# Q:Annette?
			# the default for missing data is "notAvailable"
			learner_language.learner_language_is_spoken_home = (
				meta.NaString.NOT_AVAILABLE
			)

			# default for missing data for this var is "notAvailable"
			learner_language.learner_language_parent_l1 = meta.NaString.NOT_AVAILABLE

			"""
			for three vars, we have separate columns for three languages (not necess as l1s!): fra, eng, deu

			learner_language_exposure_durationOfUse_deu
			learner_language_exposure_durationOfUse_fra
			learner_language_exposure_durationOfUse_eng

			learner_language_exposure_ageOfOnset_deu
			learner_language_exposure_ageOfOnset_fra
			learner_language_exposure_ageOfOnset_eng

			learner_language_exposure_input_deu
			learner_language_exposure_input_fra
			learner_language_exposure_input_eng

			"""

			# CSUBSUBSECTION: learner language exposure
			lle = meta.LanguageExposure()
			learner_language.exposure = lle

			try:
				curr_learner_ageProduction = float(
					lrn_meta_data_match_df.iloc[0]["text_learner_ageProduction"]
				)
			except:
				curr_learner_ageProduction = meta.NaString.NOT_AVAILABLE
			logger.info(f"Age: {curr_learner_ageProduction}")

			raw_doi = lrn_meta_data_match_df.iloc[0][
				"learner_language_exposure_durationOfInstruction_deu"
			]
			try:
				curr_duration_OfInstruction = float(raw_doi)
			except:
				curr_duration_OfInstruction = meta.NaString.NOT_AVAILABLE
			logger.info(
				f"Duration of Instruction: {curr_duration_OfInstruction} {raw_doi}"
			)
			logger.info(
				f"type check {str(type(curr_learner_ageProduction))} {str(type(curr_duration_OfInstruction))}"
			)

			# ITEM: learner_language_exposure_onset
			#
			# General idea in schema: for L2s, when this is missing , we have "notAvailable"
			# for L1s, we put 0 , no matter if the language in question is a/the target language or not
			#
			# for wtld, we look up the info for german in the learner metadata table "learner_language_exposure_ageOfOnset (deu|fra|eng)"
			# the wtld student table has values for both l1 and l2 learners of german
			#
			# The wü table says the following re the mapping:
			# curr lan == deu: if l1=deu -> "0"; IF los="f" -> age - durationOfUse; IF los = "d" OR "d_e" -> notAvailable
			# curr lan == fra: IF l1=fra -> "0"; IF l1(a,b,c)!=(fra)+los=f -> (age-schoolyear)*12;
			# curr_lan == eng: IF l1=eng -> "0"; IF l1(a,b,c)!=(eng)+los=d_e-> (age-schoolyear)*12;
			# if curr_lan anything other than deu, fra, eng : für l1a, l1b, l1c -> "0"
			#

			# for lgs other than the three project languages (fra,deu, eng), if they're L1s
			year_of_schooling = lrn_meta_data_match_df["schoolyear"].iloc[0]

			if lan_string not in PROJ_LANGS:
				if "L1" in curr_language_status_list:
					lle.learner_language_exposure_onset = 0
				else:
					lle.learner_language_exposure_onset = meta.NaString.NOT_AVAILABLE
			# project languages other than German
			elif lan_string in ["fra", "eng"]:
				# try retrievel
				value = lrn_meta_data_match_df.iloc[0][
					f"learner_language_exposure_ageOfOnset_{lan_string}"
				]
				if not pd.isna(value):
					lle.learner_language_exposure_onset = value
				# if retrieval doesn't work, try computation
				else:
					if "L1" in curr_language_status_list:
						lle.learner_language_exposure_onset = 0
					else:
						if lan_string in learners_spoken_school_language_list:
							#  (age-schoolyear)*12;
							if (
								isinstance(curr_learner_ageProduction, float)
								or isinstance(curr_learner_ageProduction, int)
								and isinstance(year_of_schooling, float)
								or isinstance(year_of_schooling, int)
							):
								lle.learner_language_exposure_onset = (
									curr_learner_ageProduction - year_of_schooling
								) * 12
							else:
								lle.learner_language_exposure_onset = (
									meta.NaString.NOT_AVAILABLE
								)
						else:
							lle.learner_language_exposure_onset = (
								meta.NaString.NOT_AVAILABLE
							)

			elif lan_string in ["deu", "ger"]:
				# try retrievel
				value = lrn_meta_data_match_df.iloc[0][
					f"learner_language_exposure_ageOfOnset_{lan_string}"
				]
				if not pd.isna(value):
					lle.learner_language_exposure_onset = value
				else:
					# if retrieval doesn't work, try computation
					if "L1" in curr_language_status_list:
						lle.learner_language_exposure_onset = 0
					elif "fra" in learners_spoken_school_language_list:
						duruse = lrn_meta_data_match_df.iloc[0][
							"learner_language_exposure_durationOfUse_deu"
						]
						if (
							isinstance(curr_learner_ageProduction, float)
							or isinstance(curr_learner_ageProduction, int)
							and isinstance(duruse, float)
							or isinstance(duruse, int)
						):
							lle.learner_language_exposure_onset = (
								curr_learner_ageProduction - duruse
							)
						else:
							lle.learner_language_exposure_onset = (
								meta.NaString.NOT_AVAILABLE
							)
					elif "deu" in learners_spoken_school_language_list:
						lle.learner_language_exposure_onset = (
							meta.NaString.NOT_AVAILABLE
						)
					else:
						logger.warning("we shouldn't really get here!")
						lle.learner_language_exposure_onset = (
							meta.NaString.NOT_AVAILABLE
						)

			# 	# REVIEW: do we want to do this?
			# 	curr_learner_ageProduction = meta.NaString.NOT_AVAILABLE
			# 	curr_duration_OfInstruction = meta.NaString.NOT_AVAILABLE

			# ITEM: learner_language_exposure_onset_group
			# not in WÜ-table,  not a learner metadatum,
			# TODO: as ofMi 11 Jun 2025 ( KW23 ) , schema excel still says: Wertebereich festlegen für onset_group; basiered auf learner_target_language_onset aus student metadata
			lle.learner_language_exposure_onset_group = meta.NaString.NOT_AVAILABLE

			# ITEM: learner_language_exposure_duration_of_instruction
			# look at column in learner metadata : learner_language_exposure_durationOfInstruction_deu
			curr_lrn_durOfinstForLang = lrn_meta_data_match_df.iloc[0][
				"learner_language_exposure_durationOfInstruction_deu"
			]

			# for all languages other than German we directly set the default value of notAvailable
			if lan_string not in ["deu", "ger"]:
				lle.learner_language_exposure_duration_of_instruction = (
					meta.NaString.NOT_AVAILABLE
				)
			# if current langauge is german
			else:
				# first try retrieval
				if not pd.isna(curr_lrn_durOfinstForLang) and lan_string in [
					"deu",
					"ger",
				]:
					lle.learner_language_exposure_duration_of_instruction = (
						curr_lrn_durOfinstForLang
					)

				# else follow rules applying to German as an L2 depending on canton
				# alle Ids mit "RI..." => 6, alle IDs mit "Mo..." und "ES..." => 5, alle anderen "notAvailable"
				# NB: the below (still) gives integers (years)!
				else:
					# Q: wouldn't it be semantically more direct to ask if German is an "L1"?
					if src_learner_id.lower() not in ["ri", "mo", "es"]:
						lle.learner_language_exposure_duration_of_instruction = (
							# AP: didn't have not applicable earlier?
							# meta.NaString.NOT_APPLICABLE
							meta.NaString.NOT_AVAILABLE
						)

					else:
						# alle Ids mit "RI..." => 6, alle IDs mit "Mo..." und "ES..." => 5, alle anderen "notApplicable"
						if src_learner_id.lower().startswith("ri"):
							lle.learner_language_exposure_duration_of_instruction = 6
						elif src_learner_id.lower().startswith("mo"):
							lle.learner_language_exposure_duration_of_instruction = 5
						elif src_learner_id.lower().startswith("es"):
							lle.learner_language_exposure_duration_of_instruction = 5
						else:
							logger.warning(
								"we should not get into this condition! there must be canton we haven't accounted for "
							)
							lle.learner_language_exposure_duration_of_instruction = (
								meta.NaString.NOT_AVAILABLE
							)

			# ITEM: learner_language_exposure_duration_of_use
			# look up in students/learner table "learner_language_exposure_durationOfUse_{lan_string}"
			# currlan==deu: if l1=deu -> age*12; IF los=f AND l1a!= deu -> learner_language_exposure_durationOfInstruction; if los=d ODER d_e AND If l1a!= deu -> schoolyear*12;
			# currlan==fra: if l1=fra -> age*12; IF los=f AND l1!=fra -> schoolyear*12; IF los!=f and l1!=f -> Sprache nicht dokumentiert
			# currlan==eng: if l1=eng -> age*12; IF los=de;eng AND l1!=eng -> schoolyear*12; IF los!=deu;eng and l1!=eng -> Sprache nicht dokumentiert
			# for all lgs other than deu/fra/enng that are L1=> age*12

			# NB: we have direct info in teh students table only for German!
			curr_lrn_durOfuseForLang = lrn_meta_data_match_df.iloc[0][
				"learner_language_exposure_durationOfUse_deu"
			]
			# if lg is not None, we use it
			if not pd.isna(curr_lrn_durOfuseForLang) and lan_string in ["deu", "ger"]:
				lle.learner_language_exposure_duration_of_use = curr_lrn_durOfuseForLang

			# else follow rules of schema
			else:
				if lan_string in ["fra", "eng"]:
					if "L1" in curr_language_status_list:
						if isinstance(curr_learner_ageProduction, float) or isinstance(curr_learner_ageProduction, int):
							lle.learner_language_exposure_duration_of_use = (
							curr_learner_ageProduction * 12
							)
						else:
							lle.learner_language_exposure_duration_of_use =  meta.NaString.NOT_AVAILABLE
					elif lan_string in learners_spoken_school_language_list:
						# NB: we're assuming that the language of instruction is in place from the beginning of school attendance
						lle.learner_language_exposure_duration_of_use = (
							year_of_schooling * 12
						)
					else:
						lle.learner_language_exposure_duration_of_use = (
							meta.NaString.NOT_AVAILABLE
						)
				elif lan_string in ["deu", "ger"]:
					# # currlan==deu: if l1=deu -> age*12;

					if "L1" in curr_language_status_list:
						lle.learner_language_exposure_duration_of_use = (
							curr_learner_ageProduction * 12
						)
					# IF los=f AND l1a!= deu -> learner_language_exposure_durationOfInstruction;
					# NB: we take duration of instruction as a proxy for duration of use!
					elif "fra" in learners_spoken_school_language_list:
						lle.learner_language_exposure_duration_of_use = (
							lle.learner_language_exposure_duration_of_instruction
						)
					# if los=d ODER d_e AND If l1a!= deu -> schoolyear*12;
					elif "deu" in learners_spoken_school_language_list or "ger" in learners_spoken_school_language_list:
						lle.learner_language_exposure_duration_of_use = (
							year_of_schooling * 12
						)

					# IF los!=f and l1!=f -> Sprache nicht dokumentiert
					else:
						lle.learner_language_exposure_duration_of_use = (
							meta.NaString.NOT_AVAILABLE
						)
				# languages other than the three project langauges
				else:
					if "L1" in curr_language_status_list:
						if isinstance(curr_learner_ageProduction,int) or isinstance(curr_learner_ageProduction, float):
							lle.learner_language_exposure_duration_of_use = (
							curr_learner_ageProduction * 12
						)
						else:
							lle.learner_language_exposure_duration_of_use = meta.NaString.NOT_AVAILABLE
					else:
						lle.learner_language_exposure_duration_of_use = (
							meta.NaString.NOT_AVAILABLE
						)

			# ITEM: learner_language_exposure_input

			# curr_lan == deu: if l1=deu ODER IF los=(d|d_e) - -> "mainly without controlled teaching processes" ; IF los=f => "mainly in controlled teaching contexts" IF ELSE
			# currlan == fra: if l1=fra ODER IF los=f -> "mainly without controlled teaching processes" ; IF ELSE fra als language nicht vorhanden
			# curr_lan == eng: if l1=eng ODER IF los=d_e-> "mainly without controlled teaching processes" ; IF ELSE eng als language nicht vorhanden
			# curr_lan not in PROJECT_LANGS: "mainly without controlled teaching processes"

			# we start with the non-proj langs
			if lan_string not in PROJ_LANGS:
				# we make a default assumption about L1s
				if "L1" in curr_language_status_list:
					lle.learner_language_exposure_input = (
						"mainly without controlled teaching processes"
					)
				# we don't know nothing about non-L1s but also we shouldn't get in there for SWIKO
				else:
					logger.warning("Well this is odd!")
					lle.learner_language_exposure_input = meta.NaString.NOT_AVAILABLE

			elif lan_string in ["eng", "fra"]:
				if (
					"L1" in curr_language_status_list
					or lan_string in learners_spoken_school_language_list
				):
					lle.learner_language_exposure_input = (
						"mainly without controlled teaching processes"
					)
				else:
					logger.warning("We shouldn't ever get here!")
					lle.learner_language_exposure_input = meta.NaString.NOT_AVAILABLE
			elif lan_string in ["deu", "ger"]:
				# try to retrieve a value for German
				curr_lrn_input_deu = lrn_meta_data_match_df.iloc[0][
					"learner_language_exposure_input_deu"
				]

				# if retrieved value is not None and current lang is german, use value from metadata table
				if not pd.isna(curr_lrn_input_deu):
					lle.learner_language_exposure_input = curr_lrn_input_deu
				else:
					# make inferences
					if (
						"L1" in curr_language_status_list
						or "deu" in learners_spoken_school_language_list
						or "ger" in learners_spoken_school_language_list
					):
						lle.learner_language_exposure_input = (
							"mainly without controlled teaching processes"
						)
					elif "fra" in learners_spoken_school_language_list:
						lle.learner_language_exposure_input = (
							"mainly in controlled teaching processes"
						)
					else:
						logger.warning("how did we get here? ")
						lle.learner_language_exposure_input = (
							meta.NaString.NOT_AVAILABLE
						)

			# ITEM: learner_language_exposure_months_spent_environment
			#
			# we try to look up a value for German
			if lan_string in [
				"deu",
				"ger",
			]:
				curr_lrn_month_spentenv_for_deu = lrn_meta_data_match_df.iloc[0][
					"learner_language_exposure_monthsSpentEnvironment_deu"
				]
				if not pd.isna(curr_lrn_month_spentenv_for_deu):
					lle.learner_language_exposure_months_spent_environment = (
						curr_lrn_month_spentenv_for_deu
					)
				else:
					# it is computed as age * 12
					if isinstance(curr_learner_ageProduction, int) or isinstance(
						curr_learner_ageProduction, float
					):
						inferred_months = 12 * curr_learner_ageProduction
						lle.learner_language_exposure_months_spent_environment = (
							inferred_months
						)
					else:
						lle.learner_language_exposure_months_spent_environment = (
							meta.NaString.NOT_AVAILABLE
						)
			else:
				# this value is typically uncalculable if it's not given explicitly
				lle.learner_language_exposure_months_spent_environment = (
					meta.NaString.NOT_AVAILABLE
				)

			# ITEM: learner_language_exposure_learning_context

			# info in WÜ table
			# curr_lan == deu: if l1=deu=f -> {0-text_learner_ageProduction: naturalistic}; if los=f -> {learner_language_exposure_ageOfOnset-text_learner_ageProduction: instructed}; IF ELSE -> notAvailabe
			# curr_lan == fra : "notAvailable"
			# curr_lan == eng: "notAvailable"
			# curr_lan not in PROJECT_LANGS: "notAvailable"

			# we have info / can calculate info only for German
			if lan_string in ["deu", "ger"]:
				if "L1" in curr_language_status_list:
					lle.learner_language_exposure_learning_context = (
						f"0-{curr_learner_ageProduction}: naturalistic"
					)
				elif "fra" in curr_students_school_lg:
					lle.learner_language_exposure_learning_context = f"{lle.learner_language_exposure_onset}-{curr_learner_ageProduction}: instructed"
				else:
					lle.learner_language_exposure_learning_context = (
						meta.NaString.NOT_AVAILABLE
					)
			else:
				lle.learner_language_exposure_learning_context = (
					meta.NaString.NOT_AVAILABLE
				)

			# ITEM: learner_language_exposure_place_acquisition
			# in theory applicable to L1 and L2
			# WÜ says  "CHE" for languages "deu" and "fra"
			if lan_string in ["deu", "ger", "fra"]:
				lle.learner_language_exposure_place_acquisition = "CHE"

			else:
				# for wtld/swiko we have no info
				lle.learner_language_exposure_place_acquisition = (
					meta.NaString.NOT_AVAILABLE
				)

			# ITEM: learner_language_exposure_was_in_environment (deu)
			#
			# students metadata able has column "learner_language_exposure_WasInEnvironment"
			# in Swiko, all students except those from canton VD have a value of yes/true for this variable:
			# "Alle anderen Kantone haben als Amtssprache auch Deutsch."
			#
			# if value is not None and current lang is german, use it
			curr_lrn_was_in_env = map_to_boolean(lrn_meta_data_match_df.iloc[0][
				"learner_language_exposure_WasInEnvironment_deu"
			])
			if not pd.isna(curr_lrn_was_in_env) and lan_string in ["deu", "ger"]:
				lle.learner_language_exposure_was_in_environment = curr_lrn_was_in_env
			else:
				# For other cases, we use no principled general way to infer this for SWIKO or other languages.
				#
				# E.g for students with fra as their language of schooling, we might infer that they are
				# in a french speaking environment for this to count
				if canton == "VD":
					lle.learner_language_exposure_was_in_environment = False
				# we add the case that canton is unknown
				elif canton == "notAvailable":
					lle.learner_language_exposure_was_in_environment = (
						meta.NaString.NOT_AVAILABLE
					)
				else:
					lle.learner_language_exposure_was_in_environment = True

			# ITEM:  learner_language_was_instructed
			#
			# schema says this is yes,  no for Target languages (either l1 or l2)
			#
			# the corpus table has no info but the learner table has auxiliary variable learner language exposure input
			# we condition the boolean instructed variable on the aux variable
			# (which e.g. has the values `mainly without controlled teaching processes' )
			#
			# learner_language_exposure_input="mainly in controlled teaching contexts" -> "yes";
			# if else -> "no"
			#
			# @AP: i think we put down values only for German, is that right?
			if (
				lan_string in ["deu", "ger"]
				and "Target language" in curr_language_status_list
			):
				#  retrieval value from students table
				llexp_input = lrn_meta_data_match_df.iloc[0][
					"learner_language_exposure_input_deu"
				]
				# map to boolean
				if "mainly in" in llexp_input:
					lle.learner_language_was_instructed = True

				elif "mainly without" in llexp_input:
					lle.learner_language_was_instructed = False
				# if it can't be mapped to boolean, it must be notAvailable
				else:
					lle.learner_language_was_instructed = meta.NaString.NOT_AVAILABLE
			# the concept is not applicable to non-target languages
			elif not "Target language" in curr_language_status_list:
				lle.learner_language_was_instructed = meta.NaString.NOT_APPLICABLE
			# we do nothing about target languages other than German if they exist in the data
			else:
				lle.learner_language_was_instructed = meta.NaString.NOT_AVAILABLE

			# CSUBSUBSECTION: learner language proficiency

			# For SWIKO most of this info is not available basically.
			# For a few variables such as learner_language_proficiency_estimateMax , we have specific default values.
			#
			# For later reuse of this code, we already start on distinguishing
			# the full set of combinations of L1/L2 and Target language status.

			# Q: in the WÜ-Table, the assignment related variables have "alleSprachen"
			# in column "L1,L2,Target,nonTarget, iso". But I think for the other vars that have a blank in that column,
			# we actually assume relevant for all languages, right?

			llp = meta.LearnerLanguageProficiency()
			learner_language.proficiency = llp

			# ITEM: learner_language_proficiency_score
			# ITEM: learner_language_proficiency_cefr
			# ITEM: learner_language_proficiency_c_test_cefr
			# ITEM: learner_language_proficiency_c_test_percent
			# ITEM: learner_language_proficiency_c_test_level_detail
			# ITEM: learner_language_proficiency_c_test_type

			# Target L2s
			if (
				"L2" in curr_language_status_list
				and "Target language" in curr_language_status_list
			):
				# German as Target L2
				# we need to single this out because of the estimates for proficiency with learners of German as l2-target
				if lan_string in ["deu", "ger"]:
					llp.learner_language_proficiency_score = (
						default_learner_language_proficiency_score
					)
					llp.learner_language_proficiency_cefr = (
						default_learner_proficiency_Cefr
					)
					llp.learner_language_proficiency_c_test_cefr = (
						default_learner_language_proficiency_cTestCefr
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
				# l2 target languages other than German
				else:
					llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
					llp.learner_language_proficiency_cefr = meta.NaString.NOT_AVAILABLE
					llp.learner_language_proficiency_c_test_cefr = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_c_test_percent = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_c_test_level_detail = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_c_test_type = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_estimate_min = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_estimate_max = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_self_assessment = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_assignment_instrument = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_assignment_method = (
						meta.NaString.NOT_AVAILABLE
					)
					llp.learner_language_proficiency_documentation = (
						meta.NaString.NOT_AVAILABLE
					)

			# target l1s
			elif (
				"L1" in curr_language_status_list
				and "Target language" in curr_language_status_list
			):
				# for non-target languages we are also ignorant
				llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_cefr = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_c_test_cefr = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_percent = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_level_detail = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_type = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_min = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_max = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_self_assessment = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_instrument = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_method = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_documentation = (
					meta.NaString.NOT_AVAILABLE
				)

			elif (
				# non-target l1s
				#
				# Schema says we should necessarily always get notAvailable for the values
				#
				"L1" in curr_language_status_list
				and not "Target language" in curr_language_status_list
			):
				llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_cefr = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_c_test_cefr = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_percent = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_level_detail = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_type = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_min = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_max = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_self_assessment = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_instrument = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_method = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_documentation = (
					meta.NaString.NOT_AVAILABLE
				)

			elif (
				"L2" in curr_language_status_list
				and not "Target language" in curr_language_status_list
			):
				llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_cefr = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_c_test_cefr = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_percent = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_level_detail = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_type = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_min = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_max = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_self_assessment = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_instrument = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_method = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_documentation = (
					meta.NaString.NOT_AVAILABLE
				)

			else:
				llp.learner_language_proficiency_score = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_cefr = meta.NaString.NOT_AVAILABLE
				llp.learner_language_proficiency_c_test_cefr = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_percent = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_level_detail = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_c_test_type = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_min = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_estimate_max = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_self_assessment = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_instrument = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_assignment_method = (
					meta.NaString.NOT_AVAILABLE
				)
				llp.learner_language_proficiency_documentation = (
					meta.NaString.NOT_AVAILABLE
				)

		# CSECTION: TEXT related info
		text_block = meta.TextProperties()
		learner_metadata_doc.text = text_block

		# Q: is xml or csv more "canonical"? or should this be our verison
		text_block.text_file = docid + ".xml"

		# TODO: what is the dakoda id?
		text_block.text_id = meta.NaString.NOT_AVAILABLE

		# @AP: is there a specific text id, like the filename minus the extension?
		text_block.text_id_orig = meta.NaString.NOT_AVAILABLE

		# TODO: we need to create an instance of meta.Language here  from the three letter lang code
		txt_lg_name_en, txt_lg_name_de , txt_lg_group = get_lan_info_for_iso_code(
			text_lg, lang_info_df
		)

		text_lang = meta.Language(
			name_en=txt_lg_name_en, value=txt_lg_name_de, iso_code_639_3=lan_string, group=txt_lg_group
		)
		text_block.text_language = text_lang

		# Q: we get the int part of the task id ? or do we only consider the tasks that the speaker has actually completed?
		# the task int is taken out of the filename
		text_block.text_longitudinal_order = task_int

		# TODO: check if we do have this info after all:  Werteübersetzung says the students metadata should have info
		text_block.text_time_of_creation = lrn_meta_data_match_df["yearcollected"].iloc[
			0
		]

		# CURSOR
		doc_df = doc_stats_df[(doc_stats_df["docid"] == docid+".xmi" ) & (doc_stats_df["viewname"] == "ctok" ) ]
		if len(doc_df) == 1:
			docrow = doc_df.iloc[0]
			
			text_block.text_token_count = docrow["Token"]
			text_block.text_clause_count = docrow["Sentence"]

		else:
			logger.error( f"docid {docid}.xmi not found in doc_stats_df")
			text_block.text_token_count = 1234567890 # docrow["Token"]
			text_block.text_clause_count = 1234567890 # docrow["Sentence"]


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

		learner_metadata_doc.annotation = []
		spacy_anno_block = meta.Annotation()
		learner_metadata_doc.annotation.append(spacy_anno_block)
		spacy_anno_block.annotation_automatic = True
		spacy_anno_block.annotation_corrected = False
		# TODO: should this next be a url?
		spacy_anno_block.annotation_documentation = "https://spacy.io/"
		spacy_anno_block.annotation_evaluation = meta.NaString.NOT_AVAILABLE
		# since we have multiple annotation tools, should we wrap this?
		spacy_anno_block.annotation_tool = (
			"spaCy parser (German model for TiGer dependencies)"
		)
		spacy_anno_block.annotation_tool_version = "3.7.5" #meta.NaString.NOT_AVAILABLE
		spacy_anno_block.annotation_model_version = "de_dep_news_trf (3.7.2)" #meta.NaString.NOT_AVAILABLE
		spacy_anno_block.annotation_type = (
			"UPOS,XPOS,lemma,dependency_head,dependency_relation"
		)

		syntaxdot_anno_block = meta.Annotation()
		# learner_metadata_doc.annotation = []
		learner_metadata_doc.annotation.append(syntaxdot_anno_block)
		syntaxdot_anno_block.annotation_automatic = True
		syntaxdot_anno_block.annotation_corrected = False
		# TODO: should this next be a url?
		syntaxdot_anno_block.annotation_documentation = (
			"https://github.com/tensordot/syntaxdot"
		)
		syntaxdot_anno_block.annotation_evaluation = meta.NaString.NOT_AVAILABLE
		# since we have multiple annotation tools, should we wrap this?
		syntaxdot_anno_block.annotation_tool = "syntaxdot parser (UD dependencies)"
		syntaxdot_anno_block.annotation_tool_version = "syntaxdot 0.5.0"
		syntaxdot_anno_block.annotation_model_version = "https://github.com/tensordot/syntaxdot-models/releases/download/de-ud-2021/de-ud-huge-20210307.tar.gz"
		syntaxdot_anno_block.annotation_type = (
			"UPOS,XPOS,lemma,dependency_head,dependency_relation,topological fields"
		)

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
		logger.info(f"writing{ix}-th document to {xoutpath} and {joutpath}")
		with open(xoutpath, "w", encoding="utf-8") as audi:
			audi.write(serializer.render(learner_metadata_doc))
		with open(joutpath, "w", encoding="utf-8") as jaudi:
			jaudi.write(jserializer.render(learner_metadata_doc))


if __name__ == "__main__":
	main()
