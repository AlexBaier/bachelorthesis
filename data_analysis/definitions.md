# Definitions

The rules for identifying classes and root classes in Wikidata, used
by the scripts in this repository, are defined as follows:


## Class
A class is an item, as defined by Wikidata (see [Help: Items](https://www.wikidata.org/wiki/Help:Items)).

A class cannot be the [instance of (P31)](https://www.wikidata.org/wiki/Property:P31) another item.

If an item has the [subclass of (P279)](https://www.wikidata.org/wiki/Property:P279)
property, it is a class, because it is a subclass.

This also means that an item, which has subclasses (other classes point to this item with 
[P279](https://www.wikidata.org/wiki/Property:P279)), is a class.

If an item has instances (other items point to it with [P31](https://www.wikidata.org/wiki/Property:P31)), 
it is a class.

## Root class

A root class is a class, which has no parent classes.

This means, it does not have the property [subclass of (P279)](https://www.wikidata.org/wiki/Property:P279).