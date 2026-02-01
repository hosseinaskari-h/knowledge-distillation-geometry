import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.agents.hf_agent import HuggingFaceAgent
import logging
from scipy.spatial.distance import cdist

# A list of words to create a semantic context cloud
CONTEXT_WORDS = [
    "art", "science", "philosophy", "consciousness", "language", "meaning", "form",
    "space", "time", "being", "love", "hate", "life", "death", "human", "machine",
    "nature", "city", "water", "fire", "earth", "air", "light", "darkness",
    "sound", "silence", "mind", "body", "dream", "reality", "past", "present",
    "future", "knowledge", "wisdom", "ignorance", "power", "weakness", "joy",
    "sorrow", "fear", "courage", "order", "chaos", "creation", "destruction",
    "one", "many", "all", "nothing", "everything"
]

class SemanticConstellation:
    """
    Generates and visualizes a 'Semantic Constellation', including a calculated centroid.
    """
    def __init__(self, model_name='gpt2'):
        self.config = {'model': model_name, 'device': 'cpu'}
        try:
            self.agent = HuggingFaceAgent(name=f"constellation-{model_name}", config=self.config)
        except Exception as e:
            logging.error(f"Failed to load HuggingFaceAgent with model {model_name}: {e}")
            self.agent = None
            logging.warning("HuggingFaceAgent not loaded. Using dummy data.")

    def get_embeddings(self, texts):
        if self.agent is None:
            # Return a consistent dummy array for a given text
            return np.array([np.random.RandomState(hash(text) % (2**32 - 1)).rand(768) for text in texts])
        
        embeddings = []
        for text in texts:
            emb = self.agent.get_embeddings(text)
            embeddings.append(emb)
        return torch.stack(embeddings).detach().cpu().numpy()

    def generate_constellation_data(
        self,
        main_words: list,
        goal_word: str,
        context_words=None
    ):
        if context_words is None:
            context_words = CONTEXT_WORDS[:50]

        # 1. Get embeddings
        all_texts = main_words + [goal_word] + context_words
        all_vectors = self.get_embeddings(all_texts)
        
        main_vectors = all_vectors[:3]
        goal_vector = all_vectors[3:4]
        context_vectors = all_vectors[4:]

        # 2. Calculate Centroid from main words
        v_centroid = np.mean(main_vectors, axis=0)
        
        # 3. Find closest token to the centroid
        closest_token_text = "N/A"
        closest_token_vector = v_centroid # Default fallback
        similarity_score = 0.0

        if self.agent:
            try:
                with torch.no_grad():
                    embedding_matrix = self.agent.model.get_input_embeddings().weight.detach()
                
                v_centroid_tensor = torch.from_numpy(v_centroid).to(self.agent.device)
                
                # Normalize for cosine similarity
                if v_centroid_tensor.norm() > 0:
                    v_centroid_norm = v_centroid_tensor / v_centroid_tensor.norm()
                    embedding_matrix_norm = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)
                    
                    similarities = torch.matmul(embedding_matrix_norm, v_centroid_norm)
                    best_token_id = torch.argmax(similarities)
                    similarity_score = similarities[best_token_id].item()
                    
                    closest_token_vector = embedding_matrix[best_token_id].cpu().numpy()
                    closest_token_text = self.agent.tokenizer.decode([best_token_id])
            except Exception as e:
                logging.warning(f"Could not find closest token: {e}")

        # 4. Combine all vectors for t-SNE
        vectors_to_reduce = [
            main_vectors,
            goal_vector,
            np.expand_dims(v_centroid, axis=0),
            np.expand_dims(closest_token_vector, axis=0),
            context_vectors
        ]
        all_vectors_combined = np.concatenate(vectors_to_reduce)

        # 5. Run t-SNE in 3D
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(all_vectors_combined) - 1), n_iter=1000, init='pca')
        reduced_vectors = tsne.fit_transform(all_vectors_combined)

        # 6. Separate points
        offset = 0
        main_points = reduced_vectors[offset:offset+3]; offset += 3
        goal_point = reduced_vectors[offset:offset+1]; offset += 1
        centroid_point = reduced_vectors[offset:offset+1]; offset += 1
        closest_token_point = reduced_vectors[offset:offset+1]; offset += 1
        context_points = reduced_vectors[offset:];

        # 7. Find nearest context neighbors in 3D space
        # This is for the interactive plot's hover text
        k = 10
        
        neighbor_groups = {}
        
        def find_neighbors(target_point, context_points, context_words):
            distances = cdist([target_point], context_points)[0]
            neighbor_indices = np.argsort(distances)[:k]
            return {
                "points": context_points[neighbor_indices],
                "labels": [context_words[i] for i in neighbor_indices]
            }

        # Main words neighbors
        for i in range(len(main_words)):
            neighbor_groups[f"main_{i}"] = find_neighbors(main_points[i], context_points, context_words)
            
        # Goal neighbor
        neighbor_groups["goal"] = find_neighbors(goal_point[0], context_points, context_words)

        return {
            "main_words": main_words,
            "goal_word": goal_word,
            "main_points": main_points,
            "goal_point": goal_point,
            "centroid_point": centroid_point,
            "closest_token_point": closest_token_point,
            "closest_token_text": closest_token_text,
            "context_points": context_points,
            "context_labels": context_words,
            "neighbor_groups": neighbor_groups,
            "similarity_score": similarity_score,
        }

    def generate_constellation_plot(self, data: dict):
        """
        Generates a static 2D projection of the 3D constellation data.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
        ax.set_facecolor('black')

        # Plot context cloud (dim 1 vs 2)
        ax.scatter(data['context_points'][:, 0], data['context_points'][:, 1], c='gray', alpha=0.3, s=20, label='Semantic Context')

        # Plot main words
        ax.scatter(data['main_points'][:, 0], data['main_points'][:, 1], c='#4a90e2', s=150, label='Main Words')
        for i, label in enumerate(data['main_words']):
            ax.text(data['main_points'][i, 0] + 0.1, data['main_points'][i, 1] + 0.1, label, fontsize=12, color='white', ha='left')

        # Plot goal word
        ax.scatter(data['goal_point'][:, 0], data['goal_point'][:, 1], c='lime', s=250, marker='P', label=f'Goal: "{data["goal_word"]}"')
        ax.text(data['goal_point'][0, 0] + 0.1, data['goal_point'][0, 1], data['goal_word'], fontsize=14, color='lime', ha='left', weight='bold')

        # Plot centroid
        ax.scatter(data['centroid_point'][:, 0], data['centroid_point'][:, 1], c='yellow', s=200, marker='x', label='Centroid')

        # Plot closest token
        ax.scatter(data['closest_token_point'][:, 0], data['closest_token_point'][:, 1], c='cyan', s=150, marker='D', label=f'Closest: "{data["closest_token_text"]}"')
        ax.text(data['closest_token_point'][0, 0] + 0.1, data['closest_token_point'][0, 1] - 0.1, f'"{data["closest_token_text"]}"', fontsize=12, color='cyan', ha='left', style='italic')

        # Draw lines from main words to centroid
        for point in data['main_points']:
            ax.plot([point[0], data['centroid_point'][0, 0]], [point[1], data['centroid_point'][0, 1]], linestyle='--', color='yellow', alpha=0.6)
        
        ax.set_title("Semantic Constellation (2D Projection)", fontsize=18, color='white', pad=20)
        ax.set_xlabel("Semantic Dimension 1 (t-SNE)", fontsize=12, color='gray')
        ax.set_ylabel("Semantic Dimension 2 (t-SNE)", fontsize=12, color='gray')
        ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')

        plt.tight_layout()
        return fig