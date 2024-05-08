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

print(":P")
