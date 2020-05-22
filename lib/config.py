corpora = {
    'bach.chorales' : ('bach-js','https://github.com/anonymized/bach-371-chorales.git'),
    'bach.wtc' : ('bach-js','https://github.com/anonymized/bach-wtc.git'),
    'beethoven.piano-sonatas' : ('beethoven','https://github.com/anonymized/beethoven-piano-sonatas.git'),
    'beethoven.quartets' : ('beethoven','https://github.com/anonymized/beethoven-string-quartets.git'),
    'chopin.mazurkas' : ('chopin','https://github.com/anonymized/chopin-mazurkas.git'),
    'chopin.preludes' : ('chopin','https://github.com/anonymized/chopin-preludes.git'),
    'haydn.quartets' : ('haydn','https://github.com/anonymized/humdrum-haydn-quartets.git'),
    'hummel.preludes' : ('hummel','https://github.com/anonymized/hummel-preludes.git'),
    'joplin' : ('joplin','https://github.com/anonymized/joplin-rags.git'),
    'agricola' : ('jrp','https://github.com/anonymized/Agr.git'),
    'anonymous' : ('jrp','https://github.com/anonymized/Ano.git'),
    'brumel' : ('jrp','https://github.com/anonymized/Bru.git'),
    'busnoys' : ('jrp','https://github.com/anonymized/Bus.git'),
    'compere' : ('jrp','https://github.com/anonymized/Com.git'),
    'dufay' : ('jrp','https://github.com/anonymized/Duf.git'),
    'gaspar' : ('jrp','https://github.com/anonymized/Gas.git'),
    'isaac' : ('jrp','https://github.com/anonymized/Isa.git'),
    'japart' : ('jrp','https://github.com/anonymized/Jap.git'),
    'josquin' : ('jrp','https://github.com/anonymized/Jos.git'),
    'martini' : ('jrp','https://github.com/anonymized/Mar.git'),
    'obrecht' : ('jrp','https://github.com/anonymized/Obr.git'),
    'ockeghem' : ('jrp','https://github.com/anonymized/Ock.git'),
    'orto' : ('jrp','https://github.com/anonymized/Ort.git'),
    'pipelare' : ('jrp','https://github.com/anonymized/Pip.git'),
    'regis' : ('jrp','https://github.com/anonymized/Reg.git'),
    'rue' : ('jrp','https://github.com/anonymized/Rue.git'),
    'tinctoris' : ('jrp','https://github.com/anonymized/Tin.git'),
    'mozart.piano-sonatas' : ('mozart','https://github.com/anonymized/mozart-piano-sonatas.git'),
    'mozart.quartets' : ('mozart','https://github.com/anonymized/humdrum-mozart-quartets.git'),
    'scarlatti-d.keyboard-sonatas' : ('scarlatti-d','https://github.com/anonymized/scarlatti-keyboard-sonatas.git'),
}

piano_corpora = ['bach.wtc','beethoven.piano-sonatas','chopin.preludes','chopin.mazurkas','hummel.preludes','joplin','mozart.piano-sonatas','scarlatti-d.keyboard-sonatas']
string_corpora = ['beethoven.quartets','haydn.quartets','mozart.quartets']

rejected_scores = [
    # these pieces have many parts/voices that our models don't handle
    'josquin.Jos1808-Qui_habitat_in_adjutorio_altissimi',
    'josquin.Jos2405-Inviolata_integra_et_casta_es',
    'josquin.Jos9904-Proch_dolor',
    'josquin.Jos0603d-Missa_Lhomme_arme_super_voces_musicales-Sanctus',
    'josquin.Jos2105.2-Huc_me_sydereo--a_6',
    'brumel.Bru1008c-Missa_Et_ecce_terre_motus-Credo',
    'brumel.Bru1008b-Missa_Et_ecce_terre_motus-Gloria',
    'brumel.Bru1008a-Missa_Et_ecce_terre_motus-Kyrie',
    'brumel.Bru1008d-Missa_Et_ecce_terre_motus-Sanctus',
    'rue.Rue2012-Magnificat_Sexti_toni',
    'brumel.Bru1008e-Missa_Et_ecce_terre_motus-Agnus',
    'beethoven.quartets.quartet14-4',
    
    # these files have strange syntax that I couldn't deal with
    'rue.Rue1011b-Missa_De_sancta_cruce-Gloria',
    'dufay.Duf3022-Belle_vueillez_moy_vengier',
    'beethoven.piano-sonatas.sonata29-4',

    # these contain rhythms that I'd rather not deal with right now (5,7,9-tuplets)
    'tinctoris.Tin3006-Le_souvenir_I',
    'beethoven.quartets.quartet01-2',
    'beethoven.quartets.quartet01-4',
    'beethoven.quartets.quartet16-1',

    # missing parts
    'josquin.Jos1601-Deus_in_adiutorium_meum',
    'josquin.Jos2004-Magnificat_Quarti_toni-Verse_6_Fecit_potentiam',
    'orto.Ort2010-Descendi_in_ortum_meum',
]

# some corpora aren't good about specifying instruments, so here are some defaults
default_instruments = {
    'bach.wtc' : 'piano',
    'mozart.piano-sonatas' : 'piano', 
    'beethoven.piano-sonatas' : 'piano',
    'chopin.preludes' : 'piano',
    'chopin.mazurkas' : 'piano',
    'ockeghem' : 'voice',
}

test_ids = [\
    'orto.Ort1005c-Missa_Mi_mi-Credo-Missa_Petite_camusette',
    'orto.Ort1003b-Missa_La_belle_se_siet-Gloria',
    'compere.Com3033-Reveille_toy_franc_cuer_joyeulx',
    'josquin.Jos2915-Mala_se_nea',
    'rue.Rue1014d-Missa_de_septem_doloribus-Sanctus',
    'josquin.Jos0301c-Missa_Ave_maris_stella-Credo',
    'rue.Rue1006c-Missa_Conceptio_tua-Credo',
    'bach.wtc.wtc2f23',
    'haydn.quartets.op71n1-02',
    'rue.Rue1009e-Missa_De_feria-Agnus_Dei',
    'bach.chorales.chor041',
    'josquin.Jos2919-Parfons_regretz',
    'josquin.Jos0402b-Missa_Gaudeamus-Gloria',
    'compere.Com3039-Sourdes_regretz_avironez_mon_cueur',
    'bach.wtc.wtc2p20',
    'haydn.quartets.op17n3-02',
    'mozart.quartets.k421-04',
    'bach.chorales.chor090',
    'bach.wtc.wtc2f06',
    'chopin.mazurkas.mazurka33-3',
    'mozart.piano-sonatas.sonata07-2',
    'rue.Rue1011e-Missa_De_sancta_cruce-Agnus',
    'tinctoris.Tin1001b-Missa_Lhomme_arme-Gloria',
    'busnoys.Bus3039-Maintes_femmes',
    'beethoven.piano-sonatas.sonata14-1',
    'haydn.quartets.op77n2-02',
    'bach.chorales.chor311',
    'ockeghem.Ock3021-Tant_fuz_gentement',
    'bach.chorales.chor300',
    'rue.Rue1011a-Missa_De_sancta_cruce-Kyrie',
    'bach.wtc.wtc2p16',
    'haydn.quartets.op74n1-04',
    'beethoven.piano-sonatas.sonata07-3',
    'gaspar.Gas2017d-Quam_pulchra_es_Cycle-O_pulcherrima_mulierum',
    'josquin.Jos1402-Benedicite_omnia_opera',
    'josquin.Jos1512-De_profundis_clamavi',
    'josquin.Jos1708-Judica_me_Deus',
    'beethoven.quartets.quartet11-1',
    'martini.Mar1009c-Missa_La_Martinella-Credo',
    'ockeghem.Ock3025-Departes_vous',
    'ockeghem.Ock1003b-Missa_Caput-Gloria',
    'haydn.quartets.op42-02',
    'bach.chorales.chor234',
    'haydn.quartets.op33n4-02',
    'beethoven.piano-sonatas.sonata01-3',
    'beethoven.piano-sonatas.sonata15-3',
    'bach.chorales.chor063',
    'martini.Mar2031-Domine_non_est_exaltatum',
    'mozart.quartets.k171-02',
    'rue.Rue1025d-Missa_Puer_natus_est_nobis-Sanctus',
    'mozart.quartets.k421-03',
    'bach.chorales.chor212',
    'rue.Rue1017b-Missa_Inviolata-Gloria',
    'ockeghem.Ock3008-La_despourveue',
    'bach.wtc.wtc2p11',
    'bach.chorales.chor208',
    'mozart.quartets.k160-03',
    'beethoven.quartets.quartet13-3',
    'ockeghem.Ock3012-Ma_maistresse',
    'obrecht.Obr1007a-Missa_De_Sancto_Martino-Kyrie',
    'haydn.quartets.op20n1-04',
    'busnoys.Bus3061-Vous_marchez__Marionnette',
    'bach.wtc.wtc1p17',
    'josquin.Jos0601c-Missa_Lhomme_arme_quarti_toni-Credo',
    'orto.Ort1003a-Missa_La_belle_se_siet-Kyrie',
    'japart.Jap3004-De_tous_biens_plaine',
    'bach.chorales.chor079',
    'hummel.preludes.prelude67-06',
    'mozart.quartets.k157-01',
    'bach.chorales.chor371',
    'beethoven.piano-sonatas.sonata08-1',
    'orto.Ort1001c-Missa_Dominicalis-Credo',
    'mozart.quartets.k589-03',
    'pipelare.Pip1004e-Missa_Fors_seulement-Agnus',
    'bach.chorales.chor308',
    'haydn.quartets.op50n4-04',
    'josquin.Jos2203-Qui_velatus',
    'haydn.quartets.op09n3-01',
    'josquin.Jos2504-Salve_regina',
    'mozart.piano-sonatas.sonata04-2',
    'compere.Com3030-Pleut_or_a_Dieu_que_naymasse_jamais',
    'bach.chorales.chor204',
    'mozart.piano-sonatas.sonata14-1',
    'beethoven.quartets.quartet14-7',
    'bach.wtc.wtc2f12',
    'japart.Jap3008-Hellas_quelle_est_a_mon_gre',
    'haydn.quartets.op17n6-04',
    'ockeghem.Ock1014-Credo_Village',
    'josquin.Jos1003a-Missa_Quem_dicunt_homines-Kyrie',
    'haydn.quartets.op17n3-01',
    'dufay.Duf2012-Ave_regina_celorum_III',
    'ockeghem.Ock3015-Prenez_sur_moy',
    'japart.Jap3022-Vray_dieu_damours__Sancte_Iohanes_baptista__Ora_pro_nobis',
    'josquin.Jos1913-Liber_generationis',
    'anonymous.Ano3035-Sans_jamais__Allez_a_la_fougere--Dijon_Chansonnier',
    'compere.Com3005-Chanter_ne_puis_chieux_la_mynonne',
    'rue.Rue1015b-Missa_De_virginibus-Gloria',
    'josquin.Jos1001a-Missa_Mater_patris-Kyrie',
    'compere.Com3052-Scaramella_fa_la_galla',
    'josquin.Jos0406b-Missa_Veni_sancte_spiritus-Gloria',
    'mozart.piano-sonatas.sonata08-2',
    'bach.wtc.wtc1f15',
    'beethoven.quartets.quartet06-2',
    'beethoven.quartets.quartet12-2',
    'beethoven.piano-sonatas.sonata28-4',
    'josquin.Jos1102d-Missa_La_sol_fa_re_mi-Sanctus',
    'beethoven.piano-sonatas.sonata01-1',
    'ockeghem.Ock1008c-Missa_Lhomme_arme-Credo',
    'busnoys.Bus3021-Faites_de_moy',
    'gaspar.Gas2029-Ave_regina_celorum',
    'chopin.mazurkas.mazurka07-5',
    'bach.wtc.wtc2f09',
    'josquin.Jos2611-Queramus_cum_pastoribus',
    'mozart.piano-sonatas.sonata06-3a',
    'josquin.Jos0904a-Missa_Rosina-Kyrie',
    'obrecht.Obr2010-Factor_orbis__Veni_domine',
    'josquin.Jos1406-Ecce_tu_pulchra_es',
    'josquin.Jos2814-Entre_je_suis',
    'bach.chorales.chor027',
    'rue.Rue2006-Lauda_anima_mea_dominum',
    'josquin.Jos2838-Textless_piece_3',
    'bach.chorales.chor094',
    'bach.chorales.chor122',
    'bach.chorales.chor241',
    'rue.Rue1003a-Missa_Assumpta_est_Maria-Kyrie',
    'bach.chorales.chor149',
    'beethoven.quartets.quartet09-3',
    'josquin.Jos1312-Crucifixus',
    'scarlatti-d.keyboard-sonatas.L343K434',
    'josquin.Jos0302d-Missa_Da_pacem-Sanctus',
    'bach.chorales.chor303',
    'mozart.piano-sonatas.sonata11-2',
    'bach.wtc.wtc2f17',
    'josquin.Jos0403d-Missa_Pange_lingua-Sanctus',
    'ockeghem.Ock1009b-Missa_Ma_maistresse-Gloria',
    'josquin.Jos0603e-Missa_Lhomme_arme_super_voces_musicales-Agnus',
    'dufay.Duf1005d-Missa_Se_la_face_ay_pale-Sanctus',
    'josquin.Jos2720-La_belle_se_siet',
    'bach.wtc.wtc1f17',
    'bach.chorales.chor059',
    'mozart.piano-sonatas.sonata11-3',
    'josquin.Jos0406e-Missa_Veni_sancte_spiritus-Agnus',
    'haydn.quartets.op50n4-03',
    'beethoven.piano-sonatas.sonata23-3',
    'bach.chorales.chor037',
    'rue.Rue1022d-Missa_O_salutaris_hostia-Sanctus',
    'bach.chorales.chor039',
    'josquin.Jos2730-Pensif_mari',
    'josquin.Jos0501a-Missa_Lami_Baudichon-Kyrie',
    'josquin.Jos2828-Plus_nulz_regrets',
    'joplin.figleaf',
    'rue.Rue1023b-Missa_Pascale-Gloria',
    'josquin.Jos1306-Credo_Vilayge_II',
    'bach.chorales.chor006',
    'haydn.quartets.op09n3-04',
    'josquin.Jos0903c-Missa_Di_dadi-Credo',
    'josquin.Jos2311-Ave_nobilissima_creatura',
    'bach.chorales.chor145',
    'beethoven.quartets.quartet07-4',
    'haydn.quartets.op01n0-05',
    'rue.Rue1025a-Missa_Puer_natus_est_nobis-Kyrie',
    'martini.Mar2065-Laudate-pueri',
    'haydn.quartets.op20n3-04',
    'josquin.Jos1302-Credo_De_tous_biens_playne',
    'hummel.preludes.prelude67-15',
    'japart.Jap3007-Fortuna_dun_gran_tempo',
    'ockeghem.Ock1004.2a-Missa_Cuiusvis_toni-Kyrie-on_E',
    'agricola.Agr4024-Tout_a_par_moy_I',
    'haydn.quartets.op71n1-01',
    'martini.Mar1002d-Missa_Coda_di_pavon-Sanctus',
    'rue.Rue1034-Credo_sex_vocum',
    'ockeghem.Ock1004.1a-Missa_Cuiusvis_toni-Kyrie-on_D',
    'rue.Rue1005a-Missa_Ave_sanctissima_Maria-Kyrie',
    'scarlatti-d.keyboard-sonatas.L002K384',
    'scarlatti-d.keyboard-sonatas.L302K372',
    'beethoven.quartets.quartet06-4a',
    'gaspar.Gas2017a-Quam_pulchra_es_Cycle-Quam_pulchra_es',
    'josquin.Jos1414-Stetit_autem_Salomon',
    'josquin.Jos2913-Lamye_a_tous',
    'busnoys.Bus3010-Bone_chere',
    'mozart.quartets.k458-01',
    'bach.chorales.chor139',
    'joplin.newrag',
    'haydn.quartets.op55n3-04',
    'bach.chorales.chor251',
    'bach.chorales.chor112',
    'ockeghem.Ock1004.2c-Missa_Cuiusvis_toni-Credo-on_E',
    'dufay.Duf3015-Se_la_face_ay_pale',
    'rue.Rue1018c-Missa_Ista_est_speciosa-Credo',
    'martini.Mar2040-Domini_est_terra',
    'beethoven.quartets.quartet15-5',
    'josquin.Jos3011-Se_congie_prens',
    'beethoven.piano-sonatas.sonata23-2',
    'japart.Jap3005-Et_qui_la_dira__Dieu_gard_celle',
    'martini.Mar2049-Exultet_celum_laudibus',
    'bach.chorales.chor345',
    'bach.chorales.chor279',
    'martini.Mar2096-Tibi_Christe',
    'rue.Rue1021d-Missa_O_gloriosa_domina-Sanctus',
    'dufay.Duf3077-Helas_mon_dueil_a_ce_cop_sui_je_mort',
]
