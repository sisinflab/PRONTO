
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import csv


def read_file(file_path: str):
    string_list = []
    with open(file_path, "r") as file:
        for line in file:
            string_list.append(line.strip())

    return string_list


def create_query(string_list: list, predicate: str):
    values_string = "".join(["\""+entity_string.replace("\"","\\\"")+"\""+"@en " for entity_string in string_list])
    return ("select distinct ?entity ?label ?description where {" +
            f"values ?label {{{values_string}}}."
            "?entity rdfs:label ?label."
            f"?entity {predicate} ?description."
            "filter(lang(?description)=\"en\")."
            "}")

def get_results(sparql_object: SPARQLWrapper, query: str, method=POST, return_format=JSON):
    # Set the query and return format
    sparql_object.setQuery(query)
    sparql_object.setMethod(method)
    sparql_object.setReturnFormat(return_format)

    # Execute the query
    results = sparql_object.query().convert()

    return results


def write_file(file_path: str, results):
    with open(file_path, "w", newline="") as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter="\t")

        # Write the header row
        tsvwriter.writerow(["Entity", "Label", "Description"])

        # Write the results to the TSV file
        for result in results["results"]["bindings"]:
            tsvwriter.writerow([result["entity"]["value"], result["label"]["value"], result["description"]["value"]])

    print(f"Results saved to {file_path}")

def write_sharded_results(file_path: str, results):
    with open(file_path, "w", newline="") as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter="\t")

        # Write the header row
        tsvwriter.writerow(["Entity", "Label", "Description"])
        for block in results:
            for result in block["results"]["bindings"]:
                tsvwriter.writerow([result["entity"]["value"], result["label"]["value"], result["description"]["value"]])

    print(f"Results saved to {file_path}")


def process_query(src_file, predicate, output_file, offset, sparql):
    entities = read_file(src_file)

    total_results = []
    # Create the query
    for i in range(0, len(entities), offset):
        query = create_query(entities[i:i + offset], predicate)
        total_results.append(get_results(sparql, query))
    write_sharded_results(output_file, total_results)