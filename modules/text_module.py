"""
text_module.py
TF-IDF based HSN code predictor from product description.
Returns HSN code, description, GST rate, and confidence score.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── GST Rate Mapping by HSN chapter (first 2 digits) ──────────────────────────
# Based on standard GST rate slabs (0%, 5%, 12%, 18%, 28%)
HSN_GST_RATE_MAP = {
    "01": 0,   "02": 0,   "03": 5,   "04": 0,   "05": 0,
    "06": 5,   "07": 0,   "08": 0,   "09": 5,   "10": 0,
    "11": 0,   "12": 5,   "13": 5,   "14": 5,   "15": 5,
    "16": 12,  "17": 18,  "18": 18,  "19": 18,  "20": 12,
    "21": 18,  "22": 28,  "23": 0,   "24": 28,  "25": 5,
    "26": 5,   "27": 18,  "28": 18,  "29": 18,  "30": 12,
    "31": 5,   "32": 18,  "33": 18,  "34": 18,  "35": 18,
    "36": 18,  "37": 18,  "38": 18,  "39": 18,  "40": 18,
    "41": 5,   "42": 18,  "43": 18,  "44": 12,  "45": 12,
    "46": 12,  "47": 12,  "48": 12,  "49": 12,  "50": 5,
    "51": 5,   "52": 5,   "53": 5,   "54": 5,   "55": 5,
    "56": 12,  "57": 12,  "58": 12,  "59": 12,  "60": 5,
    "61": 5,   "62": 5,   "63": 5,   "64": 18,  "65": 18,
    "66": 18,  "67": 18,  "68": 18,  "69": 18,  "70": 18,
    "71": 3,   "72": 18,  "73": 18,  "74": 18,  "75": 18,
    "76": 18,  "77": 18,  "78": 18,  "79": 18,  "80": 18,
    "81": 18,  "82": 18,  "83": 18,  "84": 18,  "85": 18,
    "86": 12,  "87": 28,  "88": 5,   "89": 5,   "90": 12,
    "91": 18,  "92": 28,  "93": 28,  "94": 28,  "95": 28,
    "96": 18,  "97": 12,  "98": 5,   "99": 18,
}


def get_gst_rate(hsn_code: str) -> str:
    hsn_str = str(hsn_code).strip()
    chapter = hsn_str[:2] if len(hsn_str) >= 2 else "99"
    rate = HSN_GST_RATE_MAP.get(chapter, 18)
    return f"{rate}%"


class TextGSTPredictor:

    def __init__(self, dataset_path: str):
        self.df = pd.read_excel(dataset_path)
        self.df["HSN_Description"] = self.df["HSN_Description"].fillna("")
        self.df["HSN_CD"] = self.df["HSN_CD"].astype(str).str.strip()

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # bigrams for better matching
            min_df=1,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["HSN_Description"])

    def predict(self, user_input: str, top_n: int = 3):
        """
        Returns top N matches with HSN code, description, GST rate, confidence.
        """
        user_vector = self.vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        # Get top N indices
        top_indices = similarity.argsort()[-top_n:][::-1]
        results = []

        for idx in top_indices:
            row = self.df.iloc[idx]
            hsn = row["HSN_CD"]
            confidence = round(float(similarity[idx]) * 100, 1)
            results.append({
                "hsn":        hsn,
                "description": row["HSN_Description"],
                "gst_rate":   get_gst_rate(hsn),
                "confidence": confidence
            })

        return results
