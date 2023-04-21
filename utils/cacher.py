import os
import pickle
from dotenv import load_dotenv

load_dotenv()


class Cacher:
    """
    A class that caches the same data in file system, so we don't have to rerun the same computation again and again.
    """

    @staticmethod
    def cache(cache_name, data):
        """
        Cache the data in file system.
        :param cache_name: the name of the cache
        :param data: the data to be cached
        """
        cache_dir = os.getenv('CACHE_DIR')
        cache_path = os.path.join(cache_dir, cache_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(cache_name):
        """
        Load the cached data from file system.
        :param cache_name: the name of the cache
        :return: the cached data, or None if the cache file is not found
        """
        cache_dir = os.getenv('CACHE_DIR')
        cache_path = os.path.join(cache_dir, cache_name)
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
