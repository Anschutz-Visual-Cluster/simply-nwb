"""
Use Cases

Create NWB -> Save
Read -> Enrich -> Save -> Possibly repeat
Read -> Analysis / Graphs


NWBSession
.new()
.enrich(RawSaccadeEyePositions(args, ...))
.save("myfile")


init(filename) -> upon init, look up a specific field for list of enrichment names, map to Enrichments subclasses
enrich(EnrichmentInstance(), exist_ok=False)
save(new_filename)
sess.list_enrichments() -> [Enrichments, ..]
sess.pull("EnrichmentName.valuename") -> dispatch to correct enrichment

Enrichments
- SaccadeEyePositions
- SaccadeTimestampsAndWaveforms
..
dependencies = ["file1.txt", PreviousEnrichment()] ?
__str()__ -> name and defined values
values = {"name": ["path", "within", "nwb"}
static discoverable enrichment_name
required_fields = [["path", "in", "nwb", "required_for_enrichment_to_run"], ..]
get_value(str, nwb_obj) -> find value in NWB
run()


TODO
Make sure cno is in Anne's nwb

---
Hey, I'm working on the data pipeline and I was wondering if you had a script or something that uses your library
starting from scratch? Anna has a notebook but it's not using the most up to date version of your library. Just need to
see what funcs you're using to port them over

----
Hey I'm designing the initial structure of the NWB pipeline library, how does this example code look?
I've broken it down into 3 use cases:
1. Create an NWB from raw "conversion script"
2. "Enrich" an existing NWB with new data, saving to a new file as a "checkpoint" NWB file
3. Pull data from the NWB for analysis / misc use


Example code
--
# 1. create_nwb.py
# This is the code to create the NWB from scratch "conversion script"
sess = NWBSession.new()
sess.enrich(RawEyeTrackingCSV("path/to/file.csv"))
# ... other enrichments for raw data if needed
sess.nwb.add_processing(...) # if an enrichment doesn't exist or you want to put specific data in you can do so directly
sess.save("path/to/mysession-raw.nwb")


# 2. label_saccades.py
sess = NWBSession("path/to/mysession-raw.nwb")
sess.enrich(LabelSaccades())  # will open gui
sess.save("path/to/mysession-labeled.nwb")


# 3. run_analysis.py
sess = NWBSession("path/to/mysession-labeled.nwb")
print(sess.values("LabelSaccades"))  # Will print out all available values you can pull
other_values = sess.nwb.processing["my_module"] ..  # Can pull directly from NWB as well
saccade_values = sess.pull("LabelSaccades.saccades")  # Pull out the data
# ... do graphing / analysis things

"""
import logging
from typing import Optional, Any
from pynwb import NWBHDF5IO

from simply_nwb import SimpleNWB
from simply_nwb.pipeline.enrichments import Enrichment
from spencer_funcs.autodiscovery import discover_wrapper

from simply_nwb.pipeline.value_mapping import NWBValueMapping


class NWBSession(object):
    def __init__(self, filename, custom_enrichments: Optional[list[type]] = None):
        """
        Create a new NWB Session object from a given nwb filename. Will automatically detect enrichments in the NWB
        and compare to available. Can pass a list of custom enrichments to load in if they're not in this library

        :param filename: filepath to the nwb file
        :param custom_enrichments: list of class types for classes inheriting the Enrichment class
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NWBSession")
        self.nwb_fp = NWBHDF5IO(filename)
        self.nwb = self.nwb_fp.read()  # TODO atexit close me

        self.__builtin_enrichments = discover_enrichments()
        if custom_enrichments is not None:
            for cust in custom_enrichments:
                if not isinstance(cust, type):
                    raise ValueError(f"Custom Enrichment {cust} passed in must be a classtype and inherit from Enrichment! Use [MyEnrichment, ..] NOT [MyEnrichment(), ..]")
                # Not going to bother to check if the object type passed is actually a subclass TODO?
                self.__builtin_enrichments[cust.get_name()] = cust

        self.__enrichments = set()  # list of str names of current enrichments in the nwb file
        # TODO crawl nwb using __builtin_enrichments and fill out __enrichments with existing ones

        for k in list(self.nwb.processing.keys()):
            if k.startswith("Enrichment."):
                self.__enrichments.add(k[len("Enrichment."):])
        tw = 2

    def available_enrichments(self):
        return list(self.__enrichments)

    def available_keys(self, enrichment_name):
        if enrichment_name not in self.__enrichments:
            raise ValueError(f"Enrichment '{enrichment_name}' not found in NWB, found '{self.available_enrichments()}'")
        return Enrichment.keys(enrichment_name, self.nwb)

    def enrich(self, enrichment: Enrichment):
        if not isinstance(enrichment, Enrichment):
            raise ValueError(f"Invalid enrichment type received! Got {type(enrichment)}")

        # TODO requirement checking, for fields that are needed for adding specific enrichments
        enrichment.run(self.nwb)
        self.__enrichments.add(enrichment.get_name())

    def pull(self, namespaced_key: str) -> Any:
        """
        Pull data from the NWB using namespaced valued from the enrichments

        :param namespaced_key: Key for the value to retrieve, namespaced. ie ExampleEnrichment.myvar
        """

        namespace = namespaced_key.split(".")[0]  # namespace, eg 'ExampleEnrichment' from 'ExampleEnrichment.myvar'
        key = ".".join(namespaced_key.split(".")[1:])  # The rest, 'myvar'

        if namespace not in self.__enrichments:
            raise ValueError(f"Enrichment '{namespace}' not found in this NWBSession! Found enrichments '{str(list(self.__enrichments))}'")

        val = self.__builtin_enrichments[namespace].get_val(namespace, key, self.nwb)
        return val

    def save(self, filename):
        v = self.nwb
        SimpleNWB.write(v, filename)


@discover_wrapper
def discover_enrichments():
    def get_enrichment_name(cls) -> dict[str, type]:
        if hasattr(cls, "get_name"):
            return cls.get_name()
        else:
            return None

    return [
        "simply_nwb.pipeline.enrichments",
        Enrichment,
        get_enrichment_name
    ]
