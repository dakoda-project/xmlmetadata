# xmlmetadata

* XML schema for Dakoda metadata.
* The xml schema version employed here is 1.1. because we use assertions for checking constraints in the data. ( A command line tool to validate against xsd 1.1. is `https://github.com/jeszy75/xsd11-validator`)
* The python code to generate xml file is based on `xsdata`.



<!-- Scripts for linearizing, restructuring and evaluating xml metadata for language corpora -->

# Schema Description
## DAKODA Metadata Scheme in tabular formats
* Repository for the DAKODA metadata scheme.
* This repository provides a public dataset as a CSV file. This file can be downloaded and used locally. 
* Excel template with formatting, table layout, and filters. This file can be used to display the CSV in Excel. 
## 
Content of the tables: 
* In addition to variable names, definitions, and value ranges in German and English, there is a comment field for further details, the definition of a default value (if information is missing), and an indication of whether multiple values are permitted. 
* In the table "DAKODA vs. LC-meta", behind the columns that refer to the DAKODA scheme, the corresponding information for variables from the LC-meta scheme (Granger & Paquot, 2017; Paquot et al., 2023;  Paquot et al., 2024) is stored with a comment if the variables from the DAKODA scheme and LC-meta differ conceptually.

## DAKODA Metadata Scheme as HTML mind maps
The mind maps group the variables into thematic and hierarchical levels. 
* Mindmap_definitions: In addition to the variable names, their definitions and value ranges can also be displayed.
* Mindmap_DAKODA_vs._LC-meta: This mind map documents parallels to LC-meta, namely whether a DAKODA variable is assigned a corresponding LC-meta variable and whether its definition differs from that of LC-meta.

For more information about the project, visit https://www.dakoda.org
