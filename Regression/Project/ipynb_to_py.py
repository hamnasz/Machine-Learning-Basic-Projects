import json
import argparse

def convert_ipynb_to_py(notebook_path, output_path):
    try:
        # Read the .ipynb file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Open the output .py file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write('# -*- coding: utf-8 -*-\n')
            f.write('# Converted from Jupyter Notebook\n\n')
            
            # Process each cell
            for cell in notebook['cells']:
                # Handle code cells
                if cell['cell_type'] == 'code':
                    # Write source code
                    for line in cell['source']:
                        f.write(line)
                    # Add newline between cells
                    f.write('\n\n')
                
                # Handle markdown cells
                elif cell['cell_type'] == 'markdown':
                    # Write markdown as comments
                    f.write('# Markdown cell:\n')
                    for line in cell['source']:
                        f.write(f'# {line}')
                    f.write('\n\n')
        
        print(f"Successfully converted {notebook_path} to {output_path}")
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .ipynb to .py file')
    parser.add_argument('input_file', help='Input .ipynb file path')
    parser.add_argument('output_file', help='Output .py file path')
    
    args = parser.parse_args()
    convert_ipynb_to_py(args.input_file, args.output_file)