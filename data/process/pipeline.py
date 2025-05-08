import importlib
import pandas as pd
import json
from data.process import *
from typing import Dict, List


class Pipeline:
    def __init__(self, config_file: str = None):
        with open(config_file, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.data_path = self.cfg["data_path"]
        self.debug = self.cfg.get("debug", False)
        self.set_percentager()
        self.set_bucketizer()
        self.set_discretizer()
        self.set_corpus_builder()

    def set_bucketizer(self):
        bktz_cls_name = self.cfg["bucketizer"]["type"]
        args = self.cfg["bucketizer"]["params"]
        self.bucketizer = bucketizers[bktz_cls_name](**args)
        print("[✔] Bucketizer set.")

    def set_percentager(self):
        pct_cls_name = self.cfg["percentager"]["type"]
        args = self.cfg["percentager"]["params"]
        self.percentager = percentagers[pct_cls_name](**args)
        print("[✔] Percentager set.")

    def set_discretizer(self):
        dsc_cls_name = self.cfg["discretizer"]["type"]
        args = self.cfg["discretizer"]["params"]
        self.discretizer = discretizers[dsc_cls_name](**args)
        self.discretizer.set_bucketizer(self.bucketizer)
        print("[✔] Discretizer set.")

    def set_corpus_builder(self):
        cb_cls_name = self.cfg["corpus_builder"]["type"]
        args = self.cfg["corpus_builder"]["params"]
        self.corpus_builder = corpus_builders[cb_cls_name](**args)
        print("[✔] Corpus builder set.")

    def process(self):
        if not all(
            [self.bucketizer, self.percentager, self.discretizer, self.corpus_builder]
        ):
            raise ValueError("All components must be set before processing.")

        print("[🔄] Starting pipeline...")
        print(f"[🔍] Loading raw data from {self.data_path}...")
        df = pd.read_pickle(self.data_path)

        print("[📊] Step 1: Converting to percentage...")
        percent_df = self.percentager.transform(df)
        if self.debug:
            print("[🔍] Debugging: Percent DataFrame head:")
            print(percent_df.head())

        print("[🔢] Step 2: Discretizing...")
        discretized_df = self.discretizer.encode_df(percent_df)
        if self.debug:
            print("[🔍] Debugging: Discretized DataFrame head:")
            print(discretized_df.head())

        print("[🧠] Step 3: Building corpus...")
        # Pass stride parameter to build_corpus_from_df
        segment_length = self.cfg["corpus_builder"]["params"].get("segment_length", 22)
        stride = self.cfg["corpus_builder"]["params"].get("stride", segment_length)
        self.corpus_builder.build_corpus_from_df(
            discretized_df, segment_length=segment_length, stride=stride
        )
        if self.debug:
            print("[🔍] Debugging: Corpus DataFrame head:")
            print(self.corpus_builder.grouped.head())

        out_file = self.cfg["name"] + ".pkl"

        print(f"[💾] Step 4: Saving to {out_file}...")
        self.corpus_builder.save_corpus(out_file)

        print("[✅] Pipeline complete.")
