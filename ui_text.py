"""Turkish UI strings for the demos.

Machine-translated with the DeepL API, then hand-reviewed against the
project glossary (the same terms used in the Turkish proposal): faz
geçişi (signal switch), kuyruk baskısı (queue pressure), ikilem bölgesi
(dilemma zone), algılama bölgesi (detection zone), gölge modu (shadow
mode). ``T`` returns the Turkish text for a known English template, or
the input unchanged.

Note: the demos render text through demo_ui.draw_text (PIL-based),
because OpenCV's Hershey fonts cannot draw Turkish characters.
"""

TR = {
    # shared
    'SPACE pause   F fullscreen   Q quit': 'BOŞLUK duraklat   F tam ekran   Q çıkış',
    'any key to close': 'kapatmak için bir tuşa basın',
    'detection zone': 'algılama bölgesi',
    'dilemma zone': 'ikilem bölgesi',
    'safe-stop line': 'güvenli duruş çizgisi',
    'GREEN': 'YEŞİL',
    'YELLOW': 'SARI',
    'RED': 'KIRMIZI',
    'WALK': 'YÜRÜ',
    'WAIT': 'BEKLE',
    'CLEAR': 'BİTİYOR',
    'PARKED': 'PARK HALİNDE',
    'All red (safety clearance)': 'Hepsi kırmızı (güvenlik aralığı)',
    # case 1
    'Cars: GREEN': 'Araçlar: YEŞİL',
    'Cars: YELLOW': 'Araçlar: SARI',
    'Walk phase': 'YÜRÜ fazı',
    'Pedestrian clearance': 'Yaya boşaltma fazı',
    'Cars approaching: {n}': 'Yaklaşan araç: {n}',
    'Peds waiting: {n}': 'Bekleyen yaya: {n}',
    'Ped wait: {v} s': 'Yaya bekleme: {v} s',
    'Walk phases served: {n}': 'Verilen YÜRÜ fazı: {n}',
    'Phase ends in: {v} s': 'Faz bitimine: {v} s',
    'Dilemma zone occupied - holding': 'İkilem bölgesi dolu - yeşil korunuyor',
    'Waiting vs fixed {c} s cycle: {a} s vs {b} s ({p}% saved)':
        'Sabit {c} s döngüye kıyasla bekleme: {a} s / {b} s ({p}% tasarruf)',
    'Case 1 - Pedestrian Crossing': 'Durum 1 - Yaya Geçidi',
    'Case 1 - Pedestrian Crossing (Real, shadow mode)':
        'Durum 1 - Yaya Geçidi (Gerçek görüntü, gölge modu)',
    'Simulated time: {v} s': 'Simülasyon süresi: {v} s',
    'Average pedestrian wait: {v} s': 'Ortalama yaya bekleme süresi: {v} s',
    'No pedestrian waiting = the cars never see red.':
        'Bekleyen yaya yoksa araçlar hiç kırmızı görmez.',
    'Frames analysed: {n}': 'Analiz edilen kare: {n}',
    'Walk phases granted: {n}': 'Verilen YÜRÜ fazı: {n}',
    'Parked cars excluded from demand (max): {n}':
        'Talebe sayılmayan park halindeki araç (en çok): {n}',
    'Real pedestrians detected; every walk began at a measured safe gap.':
        'Gerçek yayalar algılandı; her YÜRÜ fazı ölçülmüş güvenli bir boşlukta başladı.',
    # case 2
    'Road 1': 'Yol 1',
    'Road 2': 'Yol 2',
    'Road 1 has GREEN': 'Yol 1: YEŞİL',
    'Road 2 has GREEN': 'Yol 2: YEŞİL',
    'Changing over ({a} / {b})': 'Faz değişiyor ({a} / {b})',
    'Road 1: {n} cars   pressure {v}': 'Yol 1: {n} araç   kuyruk baskısı {v}',
    'Road 2: {n} cars   pressure {v}': 'Yol 2: {n} araç   kuyruk baskısı {v}',
    'Switches: {n}   elapsed: {t} s': 'Faz geçişi: {n}   süre: {t} s',
    'Waiting vs fixed timer: {a} s vs {b} s ({p}% saved)':
        'Sabit plana kıyasla bekleme: {a} s / {b} s ({p}% tasarruf)',
    'Case 2 - Two-Road Intersection': 'Durum 2 - İki Yollu Kavşak',
    'Case 2 - Two-Road Intersection (Real, shadow mode)':
        'Durum 2 - İki Yollu Kavşak (Gerçek görüntü, gölge modu)',
    'Signal switches: {n}': 'Sinyal faz geçişi: {n}',
    'Waiting time vs fixed {c} s timer: {a} s vs {b} s ({p}%)':
        'Sabit {c} s plana kıyasla bekleme: {a} s / {b} s ({p}%)',
    'Identical cars ran in an invisible twin world under a dumb fixed timer - that is the saving.':
        'Aynı araçlar görünmez bir ikiz dünyada sabit zamanlı planla sürdü - tasarruf budur.',
    'Avg vehicles/frame  Road 1: {a}   Road 2: {b}':
        'Ort. araç/kare   Yol 1: {a}   Yol 2: {b}',
    'Live YOLOv8 detection driving the adaptive plan.':
        'Uyarlanabilir planı canlı YOLOv8 algılaması yönetiyor.',
    # case 3
    'North': 'Kuzey',
    'South': 'Güney',
    'East': 'Doğu',
    'West': 'Batı',
    'North-South': 'Kuzey-Güney',
    'East-West': 'Doğu-Batı',
    '{axis} road has GREEN': '{axis} yolu: YEŞİL',
    '{axis} road: YELLOW (changing)': '{axis} yolu: SARI (faz değişiyor)',
    'Cars waiting  North:{n}  South:{s}  East:{e}  West:{w}':
        'Bekleyen araç   Kuzey:{n}  Güney:{s}  Doğu:{e}  Batı:{w}',
    'Switches: {n} (demand-driven: {m})': 'Faz geçişi: {n} (talep kaynaklı: {m})',
    'Avg wait  North-South: {a} s  East-West: {b} s':
        'Ort. bekleme   Kuzey-Güney: {a} s   Doğu-Batı: {b} s',
    'Waiting vs fixed {c} s timer: {a} s vs {b} s ({p}% saved)':
        'Sabit {c} s plana kıyasla bekleme: {a} s / {b} s ({p}% tasarruf)',
    'Case 3 - Four-Way Intersection': 'Durum 3 - Dört Yönlü Kavşak',
    'Case 3 - Four-Way Intersection (Real, shadow mode)':
        'Durum 3 - Dört Yönlü Kavşak (Gerçek görüntü, gölge modu)',
    'Demand-driven (no timer): {m} of {n}':
        'Talep kaynaklı (zamanlayıcısız): {n} geçişin {m} tanesi',
    'An empty road never holds a green hostage.':
        'Boş bir yol yeşil ışığı asla rehin tutmaz.',
    'SHADOW MODE': 'GÖLGE MODU',
    'switches {n} (demand {m})': 'faz {n} (talep {m})',
    'parked ignored {n}': 'park halinde (sayılmadı) {n}',
    'cars in zone {n}': 'bölgedeki araç {n}',
    'peds {n}': 'yaya {n}',
    'cars': 'araçlar',
    'ped': 'yaya',
    'Real detections, adaptive plan overlaid - no signal hardware touched.':
        'Gerçek algılamalar, uyarlanabilir plan üstte - sinyal donanımına dokunulmadı.',
}


def T(text: str) -> str:
    """Translate a UI template; unknown strings pass through unchanged."""

    return TR.get(text, text)
