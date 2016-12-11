# Results

Analysis was executed on the Wikidata JSON dump of 2016-11-07.

**28681** root classes were found in **1,308,739** classes.

Out of these **28681** root classes, ...

* **26111** have an English label.
* **11603** have an English or Simple English Wikipedia article.

---

* **3.698** properties per class on average.
* **2029** have no properties.
* **3041** have one property.
* **23611** have more than one property.

---

* **2.64** instances per class on average.
* **17498** have no instances.
* **6722** have one instance.
* **4461** have more than one instance.

---

* **6.06** subclasses per class on average.
* **10010** have no subclasses.
* **6012** have one subclass.
* **12659** have more than one subclass.

The 20 most frequent properties were ...

1. [instance of (P31)](https://www.wikidata.org/wiki/Property:P31) with **21192** occurrences
2. [InterPro ID (P2926)](https://www.wikidata.org/wiki/Property:P2926) with **12326** occurrences.
3. [Freebase ID (P646)](https://www.wikidata.org/wiki/Property:P646) with **7256** occurrences.
4. [topic's main category (P910)](https://www.wikidata.org/wiki/Property:P910) with **6248** occurrences.
5. [Commons category (P373)](https://www.wikidata.org/wiki/Property:P373) with **6211** occurrences.
6. [has as part (P527)](https://www.wikidata.org/wiki/Property:P527) with **4030** occurrences.
7. [image (P18)](https://www.wikidata.org/wiki/Property:P18) with **2403** occurrences.
8. [part of (P361)](https://www.wikidata.org/wiki/Property:P361) with **2095** occurrences.
9. [GND ID (P227)](https://www.wikidata.org/wiki/Property:P227) with **1997** occurrences.
10. [country (P17)](https://www.wikidata.org/wiki/Property:P17) with **1578** occurrences.

Bar graphs for all root classes:


![instance sum]


![subclass sum]


![property sum]


![property frequency]


Bar graphs ignoring unlabeled root classes:


![labeled_instance_sum]


![labeled_property_sum]


![labeled_subclass_sum]


![labeled_property_frequency_sum] 

## Observations
In comparison to the results of 2016-11-07, the number of root classes increased by ~12,000.
At the same time about ~12,000 root classes with the [InterPro ID (P2926)](https://www.wikidata.org/wiki/Property:P2926)
property can be noticed. This property is now rank 2 in the property frequency rating, while
it was not even in the top 20 before. All this "new" classes are related to chemistry, e.g. proteins.
That almost half of the root classes belong into this group for the current analysis, skews
the results massively. The "new" classes seem to have many subclasses but few instances,
in comparison to the "old" classes.



[instance sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/instance_sum_20161205.png
[property sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/property_sum_20161205.png
[subclass sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/subclass_sum_20161205.png
[property frequency]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/property_frequency_20161205.png

[labeled_instance_sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/labeled_instance_sum_20161205.png
[labeled_property_sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/labeled_property_sum_20161205.png
[labeled_subclass_sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/labeled_subclass_sum_20161205.png
[labeled_property_frequency_sum]: https://github.com/AlexBaier/bachelorthesis/blob/master/data_analysis/output/labeled_property_frequency_20161205.png
