from datasets.docs_dataset import DocsDataset, DbDocsDataset
from datasets.stock_dataset import StockMeta
from preprocess.preprocess_pipeline import PreprocessPipeline
from preprocess.docs_filterer import IDocsFilterer, StockNameFilterer, Word2VecSimilarFilterer
from preprocess.docs_labeler import IDocsLabeler, FutureReturnDocsLabeler
from preprocess.keyword_extractor import IKeywordExtractor, JiebaKeywordExtractor
from preprocess.vectorlizer import IVectorlizer, KeywordsTfIdfVectorlizer
from utils.data import random_split_train_val
from sklearn.linear_model import LogisticRegression
from sklearn_model_process.train_eval_model import train_eval_model, display_evaluation_result

docs_dataset = DocsDataset()
stock_meta = StockMeta("./organized_data/stock_metadata.csv")
stock = stock_meta.get_stock_by_name("統一")

clf = LogisticRegression()
preprocess_pipeline = PreprocessPipeline(
    docs_filterer=Word2VecSimilarFilterer(topn=5, white_noise_ratio=0.1),
    docs_labeler=FutureReturnDocsLabeler(s=3, threshold=5),
    keywords_extractor=JiebaKeywordExtractor(),
    vectorizer=KeywordsTfIdfVectorlizer(count_features=1000, pca_components=100)
)

# split train and val
train_dataset, val_dataset = random_split_train_val(docs_dataset, split_ratio=0.8)
# preprocess dataset
print("preprocess train dataset")
train_dataset = preprocess_pipeline.preprocess(train_dataset, stock, do_fit=True, verbose=True)
print("\n\n\n\npreprocess val dataset")
val_dataset = preprocess_pipeline.preprocess(val_dataset, stock, do_fit=False, verbose=True)
# train and validate model
result = train_eval_model(clf, train_dataset, val_dataset, verbose=True)

# display result
print("\n\n\n\n\nresult:")
display_evaluation_result(result, clf, train_dataset, val_dataset)

