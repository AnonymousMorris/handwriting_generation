import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of default
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

def render_pickle_data(pickle_path):
    """
    Render data from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file
    """
    try:
        # Check if file exists
        if not os.path.exists(pickle_path):
            print(f"Error: File {pickle_path} does not exist")
            return
            
        # Load the pickle file
        with open(pickle_path, 'rb') as f:
            print(f"Loading data from: {pickle_path}")
            raw_data = pickle.load(f)
            
        # Print basic information about the data
        print(f"Data type: {type(raw_data)}")
        if isinstance(raw_data, dict):
            print("\nKeys in the data:")
            for key in raw_data.keys():
                print(f"- {key}")
                
        if isinstance(raw_data, dict) and 'drawing' in raw_data:
            drawing = raw_data['drawing']
            print("\nProcessing drawing data...")
            print(f"Drawing data type: {type(drawing)}")
            
            # Gather all points for normalization
            all_x = []
            all_y = []
            for stroke in drawing:
                all_x.extend(stroke[0])
                all_y.extend(stroke[1])
            all_x = np.array(all_x)
            all_y = np.array(all_y)

            # Center and scale
            mean_x = np.mean(all_x)
            mean_y = np.mean(all_y)
            centered_x = all_x - mean_x
            centered_y = all_y - mean_y
            scale = max(np.max(np.abs(centered_x)), np.max(np.abs(centered_y)))
            if scale == 0:
                scale = 1  # Prevent division by zero

            # Now plot normalized strokes with pen state coloring
            plt.figure(figsize=(8, 8))
            cmap = plt.cm.viridis  # You can change this colormap if you like
            ax = plt.gca()

            for stroke in drawing:
                x_coords = (np.array(stroke[0]) - mean_x) / scale
                y_coords = (np.array(stroke[1]) - mean_y) / scale
                pen_states = np.array(stroke[2])

                # Prepare segments for LineCollection
                points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Use the average pen state for each segment for color
                segment_colors = cmap((pen_states[:-1] + pen_states[1:]) / 2)

                lc = LineCollection(segments, colors=segment_colors, linewidths=2)
                ax.add_collection(lc)

                # Scatter points colored by pen state
                sc = ax.scatter(x_coords, y_coords, c=pen_states, cmap=cmap, s=20, alpha=0.8)

            plt.axis('equal')
            plt.grid(True)
            plt.title('Normalized Drawing Visualization (Pen State Colored)')
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.colorbar(sc, label='Pen State')
            output_path = os.path.join(os.path.dirname(__file__), 'drawing_visualization_penstate.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved pen state colored drawing visualization to: {output_path}")
                        
    except Exception as e:
        print(f"Error loading or rendering data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the correct path to the pickle file
    pickle_path = os.path.join(os.path.dirname(__file__), "data", "8", "4.pkl")
    render_pickle_data(pickle_path)
