
from SPARQLWrapper import SPARQLWrapper
from utils import process_query

# DBpedia endpoint URL
endpoint_url = "https://dbpedia.org/sparql"

offset = 100

classes_src_file = "src/classes.txt"
classes_predicate = "rdfs:comment"
classes_output_file = "../data/classes.tsv"

instances_src_file = "src/instances.txt"
instances_predicate = "dbo:abstract"
instances_output_file = "../data/instances.tsv"


if __name__ == "__main__":
    # Create a SPARQL Wrapper object
    sparql = SPARQLWrapper(endpoint_url)

    process_query(classes_src_file, classes_predicate, classes_output_file, offset, sparql)
    process_query(instances_src_file, instances_predicate, instances_output_file, offset, sparql)
