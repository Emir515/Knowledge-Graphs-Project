# KnowledgeGraphPortfolio
In this repository, you can find the project created for the **Knowledge Graphs course**, named **Knowledge Graph-based Travel Destination Recommendation**.  

The project is organized into **5 folders**:  

- The **Dataset** folder contains the dataset downloaded from [this link](https://data.europa.eu/data/datasets/knyy9nivf0qoovgp1lxthw?locale=en).  
  The dataset was preprocessed to be used in **Neo4j Aura** (a cloud-based graph database). The preprocessed datasets were then pushed to this [GitHub repository](https://github.com/Emir515/Knowledge-Graphs-Project).  
  To import them into Neo4j, you can use the queries provided in the **Graph Construction Enrichment** file. This file also contains the enrichment queries applied to the graph.  

- On the constructed and enriched graph, I ran the queries provided in the **Machine Learning** folder, specifically in **ML_triple_creation.txt**.  
  The results were saved in the **data** folder to be used later by the models **GraphSAGE, NNConv, and TransE**.  

- **TransE**:  
  I started with the **train_transe.py** file, which prepares the dataset (triples), splits it into training/validation/test sets, and trains the TransE model using PyKEEN with tuned hyperparameters.  
  The trained model and embeddings are saved in the **models** folder.  
  Next, the **evaluate_transe.py** file loads the trained embeddings and computes link prediction metrics such as AUC and Average Precision, saving results to **evaluation_results.txt**.  
  The **metric_transe.py** script reads the `results.json` file produced during training and prints detailed metrics in a concise format.  
  Finally, **similarity_transe.py** uses the trained embeddings to compute cosine similarity between entities.  

- **GraphSAGE**:  
  The **GraphSAGE** folder contains all scripts for training and evaluation.  
  - **data_creation_queries.txt**: Cypher queries for exporting edges and nodes from Neo4j into CSV format.  
  - **train_graphsage.py**: Trains the GraphSAGE model on the exported graph, saving embeddings and node mappings.  
  - **evaluate_graphsage.py**: Evaluates the model using the same metrics as above.  

- **NNConv**:  
  The **NNConv** folder contains scripts for training and evaluating a neural message-passing model.  
  - **train_nnconv.py**: Trains an NNConv graph neural network on the Neo4j-exported graph, saving the model, embeddings, and node mappings.  
  - **evaluate_nnconv.py**: Evaluates the embeddings with link prediction metrics, saving results in **evaluation_results.txt**.  
  - **similarity_results.py**: Computes cosine similarity between embeddings to find the most similar nodes to a given target node, saving results in **similarity_results.txt**.  

- **Logical Queries**:  
  The **Logical Queries** folder contains **Logical Queries.txt**, which includes the Cypher queries used in Neo4j to enrich the graph with logical relationships.  
