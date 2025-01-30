from CustomRBFFeedForward import *

def replace_ffn_with_rbf_bert(model, num_kernels, kernel_name):
    """
    Replaces the feedforward layers in the BERT model with custom RBF layers.
    
    Args:
        model: The BERT model (e.g., from Hugging Face Transformers).
        num_kernels: The number of kernels to use in the RBF layer.
    """
    for i, layer in enumerate(model.bert.encoder.layer):
        # print(f"Replacing feedforward layers with RBF in layer {i}")
        
        # Extracting sizes from the original Linear layers
        in_features = layer.intermediate.dense.weight.size(1)  # Input size to the feedforward
        intermediate_features = layer.intermediate.dense.weight.size(0)  # Hidden layer size
        
        # Get the device of the original layers
        original_device = layer.intermediate.dense.weight.device

        # Replace the intermediate dense layer with RBF
        layer.intermediate.dense = CustomRBFFeedForward(
            in_features=in_features,
            out_features=intermediate_features,
            num_kernels=num_kernels, 
            kernel_name=kernel_name
        ).to(original_device)

        # Replace the output dense layer with RBF
        layer.output.dense = CustomRBFFeedForward(
            in_features=intermediate_features,
            out_features=in_features,
            num_kernels=num_kernels
        ).to(original_device)

        # print(f"RBF layers in layer {i} moved to device: {original_device}")

