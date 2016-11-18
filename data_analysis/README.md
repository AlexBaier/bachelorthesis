## Purpose
The purpose of this package is to identify and extract the root classes
from the [Wikidata JSON dump](https://dumps.wikimedia.org/wikidatawiki/entities/).
After the extraction the characteristics of the root classes are analyzed.

The results of the latest analysis will be presented in results.md.

## Analyzed characteristics
* number of classes
* number of leaf classes
* number of root classes
* number of instances per root class
* number of subclasses per root class
* common properties of root classes
* [topic's main category (P910)](https://www.wikidata.org/wiki/Property:P910]) of root classes
* TODO: Add more

## How to extract root classes from the JSON dump?
You need to download a JSON dump from [https://dumps.wikimedia.org/wikidatawiki/entities/]().
Unpack the JSON dump. (Warning: In November 2016 the file had a size of ~100GB.
In future releases it will only increase.)

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

TODO: Describe usage of statistical analysis tools, once implemented.
