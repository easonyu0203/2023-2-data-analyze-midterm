"""
Preprocessing pipeline
"""
from preprocess.keyword_extractor import DefaultKeywordExtractor
from datasets.docs_dataset import DbDocsDataset
from datasets.stock_dataset import StockMeta
from preprocess.preprocess_pipeline import PreprocessPipeline, PreprocessPipeLineConfig
from preprocess.docs_filterer import DefaultFilterer
from preprocess.docs_labeler import DefaultDocsLabeler
from preprocess.vectorlizer import TFIDFVectorlizer
from preprocess.labeled_docs_filterer import Near0returnFilterer


docs_dataset = DbDocsDataset()
stock_meta = StockMeta(stock_meta_path="./organized_data/stock_metadata.csv")
stock_name = '台積電'
stock = stock_meta.get_stock_by_name(stock_name)

pipeline_config = PreprocessPipeLineConfig(
    docs_filterer=DefaultFilterer(),
    docs_labeler=DefaultDocsLabeler(s=3),
    labeled_docs_filterer=Near0returnFilterer(threshold=0.01),
    keywords_extractor=DefaultKeywordExtractor(),
    vectorizer=TFIDFVectorlizer()
)

pipeline = PreprocessPipeline(pipeline_config)

dataset = pipeline.preprocess(docs_dataset, stock, verbose=True)