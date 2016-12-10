## Purpose
The purpose of this package is to identify and extract the root classes
from the [Wikidata JSON dump](https://dumps.wikimedia.org/wikidatawiki/entities/).
After the extraction the characteristics of the root classes are analyzed.

The results of the latest analysis will be presented in [results.md](https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/results.md).

See [definitions.md](https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/definitions.md) 
for the definitions of class and root class in context of this tool.

## How use the scripts?
You need to download a JSON dump from [https://dumps.wikimedia.org/wikidatawiki/entities/](https://dumps.wikimedia.org/wikidatawiki/entities/).
Unpack the JSON dump. (Warning: In November 2016 the file had a size of ~100GB.
In future releases the size will only increase.)

Set the file paths in `config.py` to suit your needs.
`JSON_DUMP_PATH`  should at least point to the current location
of the downloaded JSON dump.

You need to execute the scripts in the following order:

1. `reduce_json_dump.py`
2. `find_classes.py`
3. `find_root_classes.py`
4. `find_characteristics.py`

The execution of these scripts will take some time, because the code
is neither optimized nor parallel, and because the JSON dump is huge.

4 files will be created as result. Each one contains one JSON object per line.

The JSON objects in `REDUCED_JSON_DUMP_PATH ` and `REDUCED_CLASSES_JSON_DUMP_PATH`
are reduced representations of the original items. They only contain the ID, label,
and values of the [instance of (P31)](https://www.wikidata.org/wiki/Property:P31)
and [subclass of (P279)](https://www.wikidata.org/wiki/Property:P279) properties of each item.
The objects have the following format:
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

The JSON objects in `ROOT_CLASS_CHARACTERISTICS_PATH` contain for
the analysis relevant properties of each root class. These are
the Wikidata ID, the English label,
the [topics(P910)](https://www.wikidata.org/wiki/Property:P910),
the English and Simple English Wikipedia title,
all properties, all subclasses, and all instances.
The objects have the following format:
```json
{
"id": "Q123",
"label": "english label",
"enwiki": "title",
"simplewiki": "title",
 "P910": ["..."],
 "properties": ["..."],
 "subclasses": ["..."],
 "instances": ["..."]
}
```

After these extraction steps, the actual analysis can happen.

Run `analyze_characteristics.py`. The following statistics about root classes are created
and written into a JSON file:

* How many root classes were found? -> `"root class count"`
* How many classes with a certain amount of properties? -> `"property counts"`
* How many classes with a certain amount of subclasses? -> `"subclass counts"`
* How many classes with a certain amount of instances? -> `"instance counts"`
* How often do certain properties occur in the classes? -> `"property frequencies"`
* How often do certain [topics(P910)](https://www.wikidata.org/wiki/Property:P910) occur in the classes? 
-> `"topic frequencies"`

Run `create_plots.py` to get a graphic representation of the analysis results.
`"topic frequencies"` is omitted, because almost no classes share a common topic.

See [results.md](https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/results.md)
to see a summary of the analysis.
