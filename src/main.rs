use rand::prelude::*;

mod creature;
use creature::*;

mod world;
use world::*;

const BOX_HEIGHT: i16 = 128;
const BOX_WIDTH: i16 = 128;
const GENOME_SIZE: usize = 6;
const INTERNAL_NEURONS_COUNT: u8 = 2;
const CREATURE_COUNT: usize = 3000;

fn main() {
    /*
    let g = vec![
        0b1000_0001_1000_0000_1111_1111_1111_1111u32,
        0b1000_0100_1000_0110_1111_1111_1111_1111u32,
    ];
    let cr = Creature::from_chromosomes(g, 1);
    let mut wr = World::new(vec![cr], BOX_HEIGHT as usize, BOX_WIDTH as usize);
    loop {
        wr.step();

        println!("{:#?}", wr.creatures[0].position);
    }
    */

    let creatures = (0..CREATURE_COUNT * GENOME_SIZE)
        .map(|_| thread_rng().gen::<u32>())
        .collect::<Vec<_>>()
        .chunks(GENOME_SIZE)
        .map(|c| Creature::from_chromosomes(c.to_vec(), INTERNAL_NEURONS_COUNT))
        .collect::<Vec<_>>();

    let mut world = World::new(creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);

    for generation in 0..10 {
        for _ in 0..300 {
            if generation % 50 == 0 {
                //world.display();
            }
            world.step();
        }
        let killed = world.creatures.iter().filter(|c| !c.alive).count();
        let filtered = world
            .creatures
            .iter()
            .filter(|c| {
                c.alive && (c.position.x >= 20 && c.position.x <= BOX_WIDTH - 20)
                /*c.alive
                && (c.position.x - BOX_WIDTH / 2).pow(2)
                    + (c.position.y - BOX_HEIGHT / 2).pow(2)
                    >= 900*/
            })
            .count();
        let survived = world
            .creatures
            .iter()
            .filter(|c| {
                c.alive && (c.position.x < 20 || c.position.x > BOX_WIDTH - 20)
                /*c.alive
                && (c.position.x - BOX_WIDTH / 2).pow(2)
                    + (c.position.y - BOX_HEIGHT / 2).pow(2)
                    < 900*/
            })
            .collect::<Vec<_>>();

        println!(
            "{}: {} survived ({} killed, {} filtered)",
            generation,
            survived.len(),
            killed,
            filtered
        );

        let new_creatures = (0..CREATURE_COUNT)
            .map(|_| {
                (
                    thread_rng().gen::<u32>() as usize % survived.len(),
                    thread_rng().gen::<u32>() as usize % survived.len(),
                )
            })
            .map(|(p1, p2)| survived[p1].reproduce(survived[p2]))
            .collect::<Vec<_>>();

        if filtered < 100 {
            println!("==================================================");
            println!("{:#?}", survived[0]);
            println!("==================================================");
            println!("{:#?}", survived[0].neurons);
            println!("==================================================");
            println!("{:#?}", survived[0].connections);
            println!("==================================================");

            std::thread::sleep(std::time::Duration::from_secs(100000000));
        }

        world = World::new(new_creatures, BOX_HEIGHT as usize, BOX_WIDTH as usize);
    }
}

/*
impl ActionType {
    pub fn to_action_result(&self, creature: &Creature) -> ActionResult {
        match self {
            ActionType::Move(d) => ActionResult::Move(creature.position + *d),
            ActionType::MoveForward => ActionResult::Move(creature.position + creature.last_move),
            ActionType::Kill => ActionResult::Kill(creature.position + creature.last_move),
        }
    }
}
*/
