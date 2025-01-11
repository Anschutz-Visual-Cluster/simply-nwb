import os

from simply_nwb.pipeline import Enrichment, NWBSession
# from simply_nwb.pipeline.util import LazyLoadObj, load_lazy_obj


class PipelineChain(object):
    def __init__(self, enrichs: list[Enrichment], save_base_name: str, save_checkpoints=True, skip_existing=True):
        """
        Class to chain enrichments along and save checkpoints for each enrichment processed
        skips over already processed enrichments
        """

        self.basename = save_base_name
        self.enrichs = enrichs
        self.skip_exist = skip_existing
        self.save = save_checkpoints

    def run(self, sess: NWBSession):
        print(f"Starting Enrichment chain of size '{len(self.enrichs)}'")

        start_idx = 0
        needs_update = False

        names = []
        for idx, enrich in enumerate(self.enrichs):
            name = f"{self.basename}_{enrich.get_name()}.nwb"
            names.append(name)

            if self.skip_exist and os.path.exists(name) and not needs_update:
                start_idx = idx
                sess = NWBSession(names[start_idx])
                # sess = LazyLoadObj(NWBSession, names[start_idx])  # TODO Lazy load this
            else:
                if idx != 0 and self.save:
                    sess = NWBSession(names[start_idx])  # TODO same here
                needs_update = True
                sess.enrich(enrich)
                if self.save:
                    sess.save(name)
                    start_idx = idx

        # return load_lazy_obj(sess)  # TODO
        return sess
