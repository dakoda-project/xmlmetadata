# xmlmetadata

* XML schema for Dakoda metadaten.
* The xml schema version employed here is 1.1. because we use assertions for checking constraints in the data. ( A command line tool to validate against xsd 1.1. is `https://github.com/jeszy75/xsd11-validator`)
* The python code to generate xml file is based on `xsdata`.



<!-- Scripts for linearizing, restructuring and evaluating xml metadata for language corpora -->

# DAKODA Schema
## Usage
* Repository for the DAKODA metadatascheme.
* This repository provides a public dataset as a CSV file. This file can be downloaded and used locally. 
* Excel template with formatting, table layout, and filters. This file can be used to display the CSV in Excel. 
## 
In addition to variable names, definitions, and value ranges in German and English, there is a comment field for further details, the definition of a default value if information is missing, and an indication of whether multiple values are permitted. Behind the columns that refer to the DAKODA schema, the corresponding information for variables from the LC-meta schema (Granger & Paquot, 2017; Paquot et al., 2023;  Paquot et al., 2024) is stored with a comment if the variables from the DAKODA schema and LC-meta differ conceptually.
For more information about the project, visit https://www.dakoda.org
