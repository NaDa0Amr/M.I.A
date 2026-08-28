import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def display_examples(
    images_dir: str,
    hypotheses: dict,
    references: dict,
    n: int = 10,
    output_path: str = None
) -> None:
    """
    Display qualitative examples: image + generated caption + reference captions.
    
    Creates a grid of n images with their generated and reference captions.
    Saves to output_path if specified.
    """
    img_names = list(hypotheses.keys())
    
    # Select n random images from hypotheses
    if len(img_names) > n:
        img_names = random.sample(img_names, n)
    else:
        n = len(img_names)
        
    # Use 2 columns if n > 5
    cols = 2 if n > 5 else 1
    rows = (n + cols - 1) // cols
    
    # Set figure size appropriately
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    if n == 1:
        axes = [axes]
    elif rows > 1 and cols > 1:
        axes = axes.flatten()
        
    for i, img_name in enumerate(img_names):
        ax = axes[i]
        
        # Load and display the image from images_dir
        img_path = Path(images_dir) / img_name
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            
        ax.axis('off')
        
        hyp = hypotheses.get(img_name, "")
        refs = references.get(img_name, [])
        
        # Show generated caption in green
        # Show reference captions below
        text_str = f"Gen: {hyp}\n\nRefs:\n"
        for idx, ref in enumerate(refs[:3]):  # show top 3 refs to save space
            text_str += f"- {ref}\n"
            
        ax.set_title(text_str, fontsize=10, loc='left', color='black', wrap=True)
        ax.title.set_color('green')
        
    # Hide any unused subplots
    for j in range(len(img_names), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    # Save figure to output_path
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
