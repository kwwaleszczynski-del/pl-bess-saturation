"""Public figures on the Polish storage fleet for the context section (quoted, not computed; to be verified).

Each row: (item, value, source, URL). "Installed" is the nameplate power of a battery; a capacity-market
"obligation" is that power multiplied by a derating factor, which for storage has changed from auction to auction.
"""
from __future__ import annotations

CONTEXT_ROWS: list[tuple[str, str, str, str]] = [
    ("Pumped storage already in the system",
     "About 1.46 GW in the URE registers at the end of 2023 (Żarnowiec 711 MW, Porąbka-Żar 540 MW, Solina 199 MW and "
     "three small plants). It is part of the observed 2024-2026 prices; the simulated fleet comes on top.",
     "URE, Magazynowanie energii elektrycznej - Raport 2024",
     "https://www.ure.gov.pl/download/9/14751/Raportmagazynyenergii.pdf"),
    ("Capacity market, all contracts of storage (main auctions 2021-2028, additional auctions 2021-2025)",
     "9.5 GW, of which 7.1 GW existing and 0.5 GW modernised pumped storage and 1.9 GW new batteries.",
     "URE, Magazynowanie energii elektrycznej - Raport 2024",
     "https://www.ure.gov.pl/download/9/14751/Raportmagazynyenergii.pdf"),
    ("Main auction for delivery year 2028 (December 2023)",
     "33 battery storage units, 1,734 MW of capacity contracts, about 25% of all obligations of that auction "
     "(7,071 MW).",
     "URE, Magazynowanie energii elektrycznej - Raport 2024",
     "https://www.ure.gov.pl/download/9/14751/Raportmagazynyenergii.pdf"),
    ("Main auction for delivery year 2029 (December 2024)",
     "About 2.5 GW of obligations for batteries, about 4.2 GW installed (industry estimate), derating factor for "
     "storage 61.3%; clearing price 264.90 PLN/kW/year (URE final results).",
     "ESS News, 18 December 2024 (secondary)",
     "https://www.ess-news.com/2024/12/18/polish-capacity-market-auction-for-2029-catalyzes-gigawatts-of-battery-storage-projects/"),
    ("Main auction for delivery year 2030 (December 2025)",
     "Clearing price 465.02 PLN/kW/year, 6,888 MW of obligations in total (URE). Batteries: about 0.7 GW of "
     "obligations, about 5 GW installed (estimate), after the derating factor for storage was cut to 13.39%.",
     "URE, Informacja Prezesa URE nr 1/2026; batteries: ESS News, 30 December 2025 (secondary)",
     "https://www.ure.gov.pl/download/9/15928/InformacjaPrezesaUrzeduRegulacjiEnergetykiNr12026.pdf"),
    ("Storage with capacity contracts by delivery year, according to PSE",
     "About 0.17 GW in 2027, 2 GW in 2028, 4 GW in 2029 and up to 5 GW in 2030.",
     "Parkiet, 10 July 2026, quoting PSE (secondary)",
     "https://www.parkiet.com/energetyka/art44802621-magazyny-pomoga-ale-nie-rozwiaza-problemu"),
    ("First large batteries",
     "200 MW / 800 MWh in Ełk energised in June 2026; PSE expected about 1 GW of storage in the system during 2026.",
     "gramwzielone.pl, 19 June 2026 (secondary)",
     "https://www.gramwzielone.pl/?p=20363672"),
    ("Grid connection conditions for storage",
     "About 60 GW issued (PSE management, September 2025); investors declared 112 GW of battery plans for the next "
     "decade (PSE, July 2026). Conditions without the advance payment due on 30 October 2026 lapse, so the stock "
     "will shrink.",
     "gramwzielone.pl, 30 September 2025 and 21 October 2026 (secondary)",
     "https://www.gramwzielone.pl/?p=20334453"),
    ("PSE transmission development plan 2027-2036 (draft, June 2026)",
     "18.2 GW of storage in 2036 and 24 GW in 2040, next to 43.3 GW of PV in 2036.",
     "PSE presentation of the draft PRSP 2027-2036",
     "https://www.gov.pl/attachment/82762589-c07f-4d57-a40c-7d64ba39c51e"),
    ("National energy and climate plan (KPEiK 2024, WAM scenario)",
     "About 2.7 GW of batteries (utility and prosumer) in 2030 and about 8.5 GW in 2040, as cited by the Minister "
     "of Energy.",
     "Answer to parliamentary question 11057, 26 August 2025",
     "https://api.sejm.gov.pl/sejm/term10/interpellations/attachment/ATTDKYHHM/i11057-o2.pdf"),
]

CONTEXT_NOTE = ("Figures are quoted from the named public sources, read on 22 October 2026; nothing here is "
                "computed by this project. Installed power and capacity obligations differ by the derating factor, "
                "and estimates from the trade press are marked as secondary. Market rules and plans change often: "
                "verify every value against the current publication before reuse.")
