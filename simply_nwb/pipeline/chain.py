import os

from simply_nwb.pipeline import Enrichment, NWBSession
# from simply_nwb.pipeline.util import LazyLoadObj, load_lazy_obj


class _SessionContainer(object):
    def __init__(self, cls, *args):
        self.cls = cls
        self.args = args
        self._sess = None

    @staticmethod
    def from_existing(sess):
        c = _SessionContainer(None, None)
        c._sess = sess
        return c

    def get(self):
        if self._sess is None:
            self._sess = self.cls(*self.args)
        return self._sess

class PipelineChain(object):
    def __init__(self, enrichs: list[Enrichment], save_base_name: str, save_checkpoints=True, skip_existing=False):
        """
        Class to chain enrichments along and save checkpoints for each enrichment processed
        skips over already processed enrichments
        """

        self.basename = save_base_name
        self.enrichs = enrichs
        self.skip_exist = skip_existing
        self.save = save_checkpoints
        assert len(enrichs) > 0, "Must have at least one Enrichment in the chain!"

    def run(self, sess: NWBSession):
        print(f"Starting Enrichment chain of size '{len(self.enrichs)}'")
        sess = _SessionContainer.from_existing(sess)

        start_idx = 0
        needs_update = False

        names = []
        for idx, enrich in enumerate(self.enrichs[:-1]):  # Dont process the last enrichment, always redo
            name = f"{self.basename}_{enrich.get_name()}.nwb"
            names.append(name)

            if self.skip_exist and os.path.exists(name) and not needs_update:
                start_idx = idx
                sess = _SessionContainer(NWBSession, names[start_idx])
            else:
                if idx != 0 and self.save:
                    sess = _SessionContainer(NWBSession, names[start_idx])
                needs_update = True
                ss = sess.get()
                ss.enrich(enrich)
                if self.save:
                    ss.save(name)
                    start_idx = idx

        # Save last enrichment in chain
        sess.get().enrich(self.enrichs[-1])
        sess.get().save(f"{self.basename}_{self.enrichs[-1].get_name()}.nwb")
        return sess.get()
