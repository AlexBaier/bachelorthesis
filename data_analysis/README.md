## Purpose
The main purpose of this package is to extract the root classes
of the Wikidata JSON dump and then create statistics about the
root classes.

The results of the latest analysis will be presented in results.md.

## How to use?
You need to download a JSON dump from [https://dumps.wikimedia.org/wikidatawiki/entities/]().
Extract the JSON dump. Warning: In November 2016 the file had a size of ~100GB.
In future releases it will only increase.
Set the file paths in `config.py` to suit your needs.
`JSON_DUMP_PATH`  should at least point to the current location
of the downloaded JSON dump.

You need to execute the scripts in the following order:
1. `reduce_json_dump.py`
2. `find_classes.py`
3. `find_root_classes.py`

The execution of these scripts will take some time, because the code
is neither optimized nor parallel, and because the JSON dump is huge.

3 files will be created as result. Each one contains one JSON object per line.

The JSON objects in `REDUCED_JSON_DUMP_PATH ` and `REDUCED_CLASSES_JSON_DUMP_PATH`
are reduced representations of the original items in the following form:
```json
{
"id": "Q123",
"label": "en label",
"P31": ["Q234"],
"P279": ["Q134"]
}
```

The JSON objects in `ROOT_CLASSES_JSON_DUMP_PATH` are exactly the same
as in the original JSON dump, therefore they follow the format described
in [https://www.mediawiki.org/wiki/Wikibase/DataModel/JSON]().

TODO: Describe usage of statistical analysis tools, once implemented.
