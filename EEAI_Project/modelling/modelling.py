from model.MultiOutputSGDClassifier import *
from model.RFmultioutputclassifier import *
from model.MultiOutputTreesEmbedding import *
from tabulate import tabulate

def model_predict(data, df, name):
    models_to_run = [
        (MultiOutputTreesEmbedding, "Random Trees Embedding"),
        (RFMultiOutputClassifier, "Random Forest Multi-Output"),
        (MultiOutputSGDClassifier, "SGD Multi-Output")
    ]
    
    results = []
    result_summary = []  # List to store tabulated data
    
    for model_class, model_name in models_to_run:
        print("\n" + "*" * 50)
        print(f"Training {model_name}...")
        
        try:
            # Initialize model
            model = model_class(
                model_name=model_name,
                embeddings=data.get_embeddings(),
                y=data.get_type()
            )
            
            # Train and evaluate
            model.train(data)
            model.predict(data.X_test)
            
            # Store results
            result = {
                'model_name': model_name,
                'model': model,
                'predictions': model.predictions
            }
            results.append(result)
            
            # Gather metrics for tabulation
            accuracy = model.get_accuracy(data) if hasattr(model, 'get_accuracy') else 'N/A'
            accuracy = f"{accuracy:.2f} %" if isinstance(accuracy, (int, float)) else accuracy

            
            result_summary.append([model_name, accuracy])
            model.print_results(data)  
            
            print(f"{model_name} trained successfully.")
        
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    # Print all results in a table format
    headers = ["Model Name", "Accuracy"]
    print("*" * 50)
    print("Accuracy of each model:")
    print("\n" + tabulate(result_summary, headers=headers, tablefmt="grid"))
    
    return results
