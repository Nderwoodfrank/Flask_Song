from flask import Flask, render_template, jsonify, request
import pandas as pd
import pickle

app = Flask(__name__, template_folder='templates')
df = pd.read_csv('song_dataset1.csv')

with open('model_data.pkl', 'rb') as file:
    model_data = pickle.load(file)

user_item_matrix = model_data['user_item_matrix']
user_similarity_df = model_data['user_similarity_df']

def get_song_recommendations(input_songs, user_item_matrix, user_similarity_df, top_n=10):
    # Get the user preferences based on the input song
    user_preferences = user_item_matrix.iloc[:, input_songs]

    # Calculate weighted song recommendations using user similarity
    weighted_recommendations = user_similarity_df.dot(user_preferences)

    # Exclude the input song from the list of recommended songs
    weighted_recommendations = weighted_recommendations.drop(user_item_matrix.columns[input_songs], errors='ignore')

    # Get top N recommendations
    top_recommendations = weighted_recommendations.nlargest(top_n)

    return top_recommendations.index.tolist()

@app.route('/get_years', methods=['GET'])
def get_years():
    years = df['year'].unique().tolist()
    return jsonify(years)

# Endpoint to get songs for a specific year
@app.route('/get_songs/<selected_year>', methods=['GET'])
def get_songs(selected_year):
    try:
        songs = df[df['year'] == int(selected_year)]['title'].unique().tolist()
        return jsonify(songs)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_details/<selected_year>/<selected_title>', methods=['GET'])
def get_details(selected_year, selected_title):
    try:
        details = df[(df['year'] == int(selected_year)) & (df['title'] == selected_title)].iloc[0]
        return jsonify({'artist': details['artist_name'], 'release': details['release'], 'song': details['song']})
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/get_song_details/<selected_year>/<selected_title>', methods=['GET'])
def get_song_details(selected_year, selected_title):
    try:
        details = df[(df['year'] == int(selected_year)) & (df['title'] == selected_title)].iloc[0]
        return jsonify({'song': details['song']})
    except Exception as e:
        return jsonify({'error': str(e)})

# Define routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    input_song = request.form.get('relevant')
    print("Input Song:", input_song)
    song_enc = df[df['song'] == input_song]['song_enc'].iloc[0]
    recommendations = get_song_recommendations(song_enc, user_item_matrix, user_similarity_df, top_n=10)
    song_enc_to_title = dict(zip(df['song_enc'], df['title']))
    recommendation_titles = [song_enc_to_title.get(song_enc, "Title not found") for song_enc in recommendations]
    print("Recommendation Titles:", recommendation_titles)
    return render_template('recommendations.html', recommendations=recommendation_titles)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
